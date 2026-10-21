//! Batch inference state (`crate::batch`). The bytes - uploaded inputs and the
//! result files - live on disk under `<data>/batches`; this holds what is
//! needed to resume a run after any crash: one row per batch, one per item.
//!
//! Two rules make a restart exact. An item is pending until a terminal row says
//! otherwise - "in flight" is never written, so nothing has to be repaired on
//! boot. And `out_len` / `err_len` are the committed prefix of the result files:
//! a result's line is appended and synced BEFORE the transaction that marks its
//! item terminal, so whatever lies past the committed length after a crash is
//! a result no row claims, and boot truncates it away.
use super::*;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS batch_files (
            id         TEXT PRIMARY KEY,
            purpose    TEXT NOT NULL,
            filename   TEXT NOT NULL,
            bytes      INTEGER NOT NULL,
            created_ms INTEGER NOT NULL,
            path       TEXT NOT NULL,
            batch_id   TEXT
        );
        CREATE TABLE IF NOT EXISTS batches (
            id                TEXT PRIMARY KEY,
            dialect           TEXT NOT NULL,
            endpoint          TEXT NOT NULL,
            model             TEXT NOT NULL,
            input_file_id     TEXT NOT NULL DEFAULT '',
            completion_window TEXT NOT NULL,
            status            TEXT NOT NULL,
            created_ms        INTEGER NOT NULL,
            expires_ms        INTEGER NOT NULL,
            in_progress_ms    INTEGER,
            finalizing_ms     INTEGER,
            completed_ms      INTEGER,
            failed_ms         INTEGER,
            expired_ms        INTEGER,
            cancelling_ms     INTEGER,
            cancelled_ms      INTEGER,
            metadata          TEXT NOT NULL DEFAULT 'null',
            errors            TEXT NOT NULL DEFAULT 'null',
            options           TEXT NOT NULL DEFAULT '{}',
            output_file_id    TEXT,
            error_file_id     TEXT,
            total             INTEGER NOT NULL DEFAULT 0,
            succeeded         INTEGER NOT NULL DEFAULT 0,
            errored           INTEGER NOT NULL DEFAULT 0,
            canceled          INTEGER NOT NULL DEFAULT 0,
            expired           INTEGER NOT NULL DEFAULT 0,
            out_len           INTEGER NOT NULL DEFAULT 0,
            err_len           INTEGER NOT NULL DEFAULT 0,
            input_tokens      INTEGER NOT NULL DEFAULT 0,
            output_tokens     INTEGER NOT NULL DEFAULT 0,
            cached_tokens     INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens  INTEGER NOT NULL DEFAULT 0,
            webhook_state     INTEGER NOT NULL DEFAULT 0,
            webhook_attempts  INTEGER NOT NULL DEFAULT 0,
            webhook_next_ms   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS batches_created ON batches(created_ms DESC);
        CREATE INDEX IF NOT EXISTS batches_list ON batches(dialect, created_ms, id);
        CREATE TABLE IF NOT EXISTS batch_items (
            batch_id  TEXT NOT NULL,
            idx       INTEGER NOT NULL,
            custom_id TEXT NOT NULL,
            off       INTEGER NOT NULL,
            len       INTEGER NOT NULL,
            state     INTEGER NOT NULL DEFAULT 0,
            attempts  INTEGER NOT NULL DEFAULT 0,
            strikes   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (batch_id, idx)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS batch_files_created ON batch_files(created_ms, id);",
    )?;
    // Scale. A batch may hold a million requests, so the indexes are sized to
    // the work LEFT, not the work done:
    // - The pending index covers state = 0 rows only. It shrinks as a run
    //   completes, and "the next pending items" stays a short index walk at
    //   any depth into the batch.
    // - The open-batch index holds only batches the run still has to look at.
    //   Polled every tick, a scan of every batch ever made would grow without
    //   bound.
    // (A finished batch's item rows are deleted outright: `batch_items_purge`.)
    // Resumable uploads (OpenAI's Uploads API): the upload, its parts, and
    // how far a completion has assembled them - so a completion a crash or a
    // client timeout interrupted continues instead of starting over.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS batch_uploads (
            id              TEXT PRIMARY KEY,
            filename        TEXT NOT NULL,
            purpose         TEXT NOT NULL,
            mime_type       TEXT NOT NULL,
            bytes           INTEGER NOT NULL,
            created_ms      INTEGER NOT NULL,
            expires_ms      INTEGER NOT NULL,
            status          TEXT NOT NULL,
            part_order      TEXT,
            assembled       INTEGER NOT NULL DEFAULT 0,
            assembled_bytes INTEGER NOT NULL DEFAULT 0,
            file_id         TEXT
        );
        CREATE TABLE IF NOT EXISTS batch_upload_parts (
            upload_id  TEXT NOT NULL,
            part_id    TEXT NOT NULL,
            bytes      INTEGER NOT NULL,
            created_ms INTEGER NOT NULL,
            PRIMARY KEY (upload_id, part_id)
        ) WITHOUT ROWID;",
    )?;
    // Idempotency-Key: a create or upload the client retries (a reply lost
    // to a dropped connection) returns the first one instead of starting
    // the same work twice. Columns arrived after the tables shipped - the
    // usual harmless-error ALTER.
    let _ = conn.execute("ALTER TABLE batches ADD COLUMN idem_key TEXT", []);
    let _ = conn.execute("ALTER TABLE batch_files ADD COLUMN idem_key TEXT", []);
    conn.execute_batch(&format!(
        "CREATE UNIQUE INDEX IF NOT EXISTS batches_idem ON batches(idem_key)
           WHERE idem_key IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS batch_files_idem ON batch_files(idem_key)
           WHERE idem_key IS NOT NULL;
        DROP INDEX IF EXISTS batch_items_state;
        CREATE INDEX IF NOT EXISTS batch_items_pending ON batch_items(batch_id, idx) WHERE state = 0;
        CREATE INDEX IF NOT EXISTS batches_open ON batches(created_ms, id) WHERE {OPEN};"
    ))?;
    Ok(())
}

/// Item states. Pending is the only non-terminal one.
pub const ITEM_PENDING: i64 = 0;
pub const ITEM_SUCCEEDED: i64 = 1;
pub const ITEM_ERRORED: i64 = 2;
pub const ITEM_CANCELED: i64 = 3;
pub const ITEM_EXPIRED: i64 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct BatchFileRow {
    pub id: String,
    pub purpose: String,
    pub filename: String,
    pub bytes: i64,
    pub created_ms: i64,
    /// Relative to the batches root.
    pub path: String,
    /// The batch whose result this is; None for an upload.
    pub batch_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchRow {
    pub id: String,
    /// "openai" | "anthropic"
    pub dialect: String,
    pub endpoint: String,
    pub model: String,
    pub input_file_id: String,
    pub completion_window: String,
    /// validating | in_progress | finalizing | completed | failed | expired |
    /// cancelling | cancelled
    pub status: String,
    pub created_ms: i64,
    pub expires_ms: i64,
    pub in_progress_ms: Option<i64>,
    pub finalizing_ms: Option<i64>,
    pub completed_ms: Option<i64>,
    pub failed_ms: Option<i64>,
    pub expired_ms: Option<i64>,
    pub cancelling_ms: Option<i64>,
    pub cancelled_ms: Option<i64>,
    pub metadata: Value,
    pub errors: Value,
    pub options: Value,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub total: i64,
    pub succeeded: i64,
    pub errored: i64,
    pub canceled: i64,
    pub expired: i64,
    pub out_len: i64,
    pub err_len: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub reasoning_tokens: i64,
    pub webhook_state: i64,
    pub webhook_attempts: i64,
    pub webhook_next_ms: i64,
    /// The client's Idempotency-Key, if it sent one (write-only: inserts
    /// carry it, lookups go through `batch_by_idem`).
    pub idem_key: Option<String>,
}

impl BatchRow {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "failed" | "expired" | "cancelled"
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchItemRow {
    pub idx: i64,
    pub custom_id: String,
    pub off: i64,
    pub len: i64,
    pub attempts: i64,
    pub strikes: i64,
}

/// One item's new durable state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchItemUpdate {
    pub idx: i64,
    pub state: i64,
    pub attempts: i64,
    pub strikes: i64,
}

/// One group commit: the items it settles, the result files' new committed
/// lengths, and the counter deltas - applied in a single transaction.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchCommit {
    pub items: Vec<BatchItemUpdate>,
    pub out_len: i64,
    pub err_len: i64,
    pub succeeded: i64,
    pub errored: i64,
    pub canceled: i64,
    pub expired: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub reasoning_tokens: i64,
}

/// The open-batch predicate. The `batches_open` index is declared with it, and
/// SQLite only uses a partial index when the query states the same condition,
/// so both spell it from this one string.
const OPEN: &str =
    "status IN ('validating', 'in_progress', 'finalizing', 'cancelling') OR webhook_state = 1";

const BATCH_COLS: &str = "id, dialect, endpoint, model, input_file_id, completion_window, status,
    created_ms, expires_ms, in_progress_ms, finalizing_ms, completed_ms, failed_ms, expired_ms,
    cancelling_ms, cancelled_ms, metadata, errors, options, output_file_id, error_file_id, total,
    succeeded, errored, canceled, expired, out_len, err_len, input_tokens, output_tokens,
    cached_tokens, reasoning_tokens, webhook_state, webhook_attempts, webhook_next_ms";

fn json_col(r: &rusqlite::Row<'_>, i: usize) -> rusqlite::Result<Value> {
    let s: String = r.get(i)?;
    Ok(serde_json::from_str(&s).unwrap_or(Value::Null))
}

fn batch_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<BatchRow> {
    Ok(BatchRow {
        id: r.get(0)?,
        dialect: r.get(1)?,
        endpoint: r.get(2)?,
        model: r.get(3)?,
        input_file_id: r.get(4)?,
        completion_window: r.get(5)?,
        status: r.get(6)?,
        created_ms: r.get(7)?,
        expires_ms: r.get(8)?,
        in_progress_ms: r.get(9)?,
        finalizing_ms: r.get(10)?,
        completed_ms: r.get(11)?,
        failed_ms: r.get(12)?,
        expired_ms: r.get(13)?,
        cancelling_ms: r.get(14)?,
        cancelled_ms: r.get(15)?,
        metadata: json_col(r, 16)?,
        errors: json_col(r, 17)?,
        options: json_col(r, 18)?,
        output_file_id: r.get(19)?,
        error_file_id: r.get(20)?,
        total: r.get(21)?,
        succeeded: r.get(22)?,
        errored: r.get(23)?,
        canceled: r.get(24)?,
        expired: r.get(25)?,
        out_len: r.get(26)?,
        err_len: r.get(27)?,
        input_tokens: r.get(28)?,
        output_tokens: r.get(29)?,
        cached_tokens: r.get(30)?,
        reasoning_tokens: r.get(31)?,
        webhook_state: r.get(32)?,
        webhook_attempts: r.get(33)?,
        webhook_next_ms: r.get(34)?,
        idem_key: None,
    })
}

/// A UNIQUE constraint said no - for an insert keyed by Idempotency-Key, the
/// twin request got there first.
pub fn is_conflict(e: &StoreError) -> bool {
    matches!(e, StoreError::Db(rusqlite::Error::SqliteFailure(f, _))
        if f.code == rusqlite::ErrorCode::ConstraintViolation)
}

fn file_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<BatchFileRow> {
    Ok(BatchFileRow {
        id: r.get(0)?,
        purpose: r.get(1)?,
        filename: r.get(2)?,
        bytes: r.get(3)?,
        created_ms: r.get(4)?,
        path: r.get(5)?,
        batch_id: r.get(6)?,
    })
}

/// The stamp column a status transition writes, if any.
fn stamp_col(status: &str) -> Option<&'static str> {
    Some(match status {
        "in_progress" => "in_progress_ms",
        "finalizing" => "finalizing_ms",
        "completed" => "completed_ms",
        "failed" => "failed_ms",
        "expired" => "expired_ms",
        "cancelling" => "cancelling_ms",
        "cancelled" => "cancelled_ms",
        _ => return None,
    })
}

impl Store {
    pub fn batch_file_insert(&self, f: &BatchFileRow) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO batch_files (id, purpose, filename, bytes, created_ms, path, batch_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET bytes = excluded.bytes",
            params![
                f.id,
                f.purpose,
                f.filename,
                f.bytes,
                f.created_ms,
                f.path,
                f.batch_id
            ],
        )?;
        Ok(())
    }

    pub fn batch_file_get(&self, id: &str) -> Result<Option<BatchFileRow>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                "SELECT id, purpose, filename, bytes, created_ms, path, batch_id
                 FROM batch_files WHERE id = ?1",
                params![id],
                file_row,
            )
            .optional()?)
    }

    /// Newest first (or oldest with `ascending`), optionally one purpose,
    /// keyset-paged by `after` (an id from the previous page).
    pub fn batch_files_list(
        &self,
        purpose: Option<&str>,
        after: Option<&str>,
        limit: usize,
        ascending: bool,
    ) -> Result<Vec<BatchFileRow>, StoreError> {
        let conn = self.lock();
        let cursor: Option<(i64, String)> = match after {
            Some(a) => conn
                .query_row(
                    "SELECT created_ms, id FROM batch_files WHERE id = ?1",
                    params![a],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?,
            None => None,
        };
        let (cmp, ord) = if ascending {
            (">", "ASC")
        } else {
            ("<", "DESC")
        };
        let sql = format!(
            "SELECT id, purpose, filename, bytes, created_ms, path, batch_id FROM batch_files
             WHERE (?1 IS NULL OR purpose = ?1)
               AND (?2 IS NULL OR (created_ms, id) {cmp} (?2, ?3))
             ORDER BY created_ms {ord}, id {ord} LIMIT ?4"
        );
        let (c_ms, c_id) = cursor.map_or((None, None), |(m, i)| (Some(m), Some(i)));
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![purpose, c_ms, c_id, limit as i64], file_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn batch_file_delete(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.execute("DELETE FROM batch_files WHERE id = ?1", params![id])?;
        // a result file the client deleted is no longer offered by its batch
        conn.execute(
            "UPDATE batches SET output_file_id = NULL WHERE output_file_id = ?1",
            params![id],
        )?;
        conn.execute(
            "UPDATE batches SET error_file_id = NULL WHERE error_file_id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Is a still-running batch reading this file? Deleting it would strand
    /// the run, so the files route refuses.
    pub fn batch_file_in_use(&self, id: &str) -> Result<bool, StoreError> {
        Ok(self.lock().query_row(
            "SELECT EXISTS(SELECT 1 FROM batches WHERE input_file_id = ?1
               AND status IN ('validating', 'in_progress', 'finalizing', 'cancelling'))",
            params![id],
            |r| r.get(0),
        )?)
    }

    pub fn batch_insert(&self, b: &BatchRow) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO batches (id, dialect, endpoint, model, input_file_id, completion_window,
               status, created_ms, expires_ms, in_progress_ms, metadata, errors, options, total,
               idem_key)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                b.id,
                b.dialect,
                b.endpoint,
                b.model,
                b.input_file_id,
                b.completion_window,
                b.status,
                b.created_ms,
                b.expires_ms,
                b.in_progress_ms,
                b.metadata.to_string(),
                b.errors.to_string(),
                b.options.to_string(),
                b.total,
                b.idem_key
            ],
        )?;
        Ok(())
    }

    /// The batch created under this Idempotency-Key, if any.
    pub fn batch_by_idem(&self, key: &str) -> Result<Option<BatchRow>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                &format!("SELECT {BATCH_COLS} FROM batches WHERE idem_key = ?1"),
                params![key],
                batch_row,
            )
            .optional()?)
    }

    /// The file uploaded under this Idempotency-Key, if any.
    pub fn batch_file_by_idem(&self, key: &str) -> Result<Option<BatchFileRow>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                "SELECT id, purpose, filename, bytes, created_ms, path, batch_id
                 FROM batch_files WHERE idem_key = ?1",
                params![key],
                file_row,
            )
            .optional()?)
    }

    /// Record an upload's Idempotency-Key once its row exists. False when the
    /// key is already taken (a concurrent twin won the race).
    pub fn batch_file_set_idem(&self, id: &str, key: &str) -> Result<bool, StoreError> {
        match self.lock().execute(
            "UPDATE batch_files SET idem_key = ?2 WHERE id = ?1",
            params![id, key],
        ) {
            Ok(_) => Ok(true),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn batch_get(&self, id: &str) -> Result<Option<BatchRow>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                &format!("SELECT {BATCH_COLS} FROM batches WHERE id = ?1"),
                params![id],
                batch_row,
            )
            .optional()?)
    }

    /// One dialect's batches, newest first. `after` pages towards older ones
    /// (both vendors' default), `before` towards newer ones (Anthropic's
    /// `before_id`).
    pub fn batch_list(
        &self,
        dialect: &str,
        after: Option<&str>,
        before: Option<&str>,
        limit: usize,
    ) -> Result<Vec<BatchRow>, StoreError> {
        let conn = self.lock();
        let key = |id: Option<&str>| -> Result<Option<(i64, String)>, StoreError> {
            Ok(match id {
                Some(i) => conn
                    .query_row(
                        "SELECT created_ms, id FROM batches WHERE id = ?1",
                        params![i],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?,
                None => None,
            })
        };
        let rows = if let Some((ms, id)) = key(before)? {
            // nearest-newer first, then flipped back to newest-first
            let mut stmt = conn.prepare(&format!(
                "SELECT {BATCH_COLS} FROM batches WHERE dialect = ?1
                   AND (created_ms, id) > (?2, ?3)
                 ORDER BY created_ms ASC, id ASC LIMIT ?4"
            ))?;
            let mut v: Vec<BatchRow> = stmt
                .query_map(params![dialect, ms, id, limit as i64], batch_row)?
                .collect::<Result<_, _>>()?;
            v.reverse();
            v
        } else {
            let (ms, id) = key(after)?.map_or((None, None), |(m, i)| (Some(m), Some(i)));
            let mut stmt = conn.prepare(&format!(
                "SELECT {BATCH_COLS} FROM batches WHERE dialect = ?1
                   AND (?2 IS NULL OR (created_ms, id) < (?2, ?3))
                 ORDER BY created_ms DESC, id DESC LIMIT ?4"
            ))?;
            stmt.query_map(params![dialect, ms, id, limit as i64], batch_row)?
                .collect::<Result<_, _>>()?
        };
        Ok(rows)
    }

    /// Every batch that still has work or a pending webhook, oldest first -
    /// the dispatcher's order of service.
    pub fn batch_open(&self) -> Result<Vec<BatchRow>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {BATCH_COLS} FROM batches WHERE {OPEN} ORDER BY created_ms ASC, id ASC"
        ))?;
        let rows = stmt.query_map([], batch_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Move a batch to `status`, stamping the transition's timestamp. A
    /// terminal status never moves again (a cancel racing a completion keeps
    /// whichever landed first); returns whether the row changed.
    pub fn batch_set_status(&self, id: &str, status: &str) -> Result<bool, StoreError> {
        let guard = "status NOT IN ('completed', 'failed', 'expired', 'cancelled')";
        let conn = self.lock();
        let n = match stamp_col(status) {
            Some(c) => conn.execute(
                &format!(
                    "UPDATE batches SET status = ?2, {c} = COALESCE({c}, ?3) WHERE id = ?1 AND {guard}"
                ),
                params![id, status, now_ms()],
            )?,
            None => conn.execute(
                &format!("UPDATE batches SET status = ?2 WHERE id = ?1 AND {guard}"),
                params![id, status],
            )?,
        };
        Ok(n > 0)
    }

    /// Move a batch from exactly `from` to `to` (stamped like
    /// [`Store::batch_set_status`]); false when it was no longer in `from`.
    pub fn batch_transition(&self, id: &str, from: &str, to: &str) -> Result<bool, StoreError> {
        let conn = self.lock();
        let n = match stamp_col(to) {
            Some(c) => conn.execute(
                &format!(
                    "UPDATE batches SET status = ?3, {c} = COALESCE({c}, ?4) WHERE id = ?1 AND status = ?2"
                ),
                params![id, from, to, now_ms()],
            )?,
            None => conn.execute(
                "UPDATE batches SET status = ?3 WHERE id = ?1 AND status = ?2",
                params![id, from, to],
            )?,
        };
        Ok(n > 0)
    }

    pub fn batch_set_failed(&self, id: &str, errors: &Value) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE batches SET status = 'failed', failed_ms = ?2, errors = ?3
             WHERE id = ?1 AND status NOT IN ('completed', 'failed', 'expired', 'cancelled')",
            params![id, now_ms(), errors.to_string()],
        )?;
        Ok(())
    }

    pub fn batch_set_total(&self, id: &str, total: i64) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE batches SET total = ?2 WHERE id = ?1",
            params![id, total],
        )?;
        Ok(())
    }

    /// Register the result files and close the batch in one transaction, so a
    /// crash leaves either the finalizing batch (redone on boot) or the
    /// finished one - never a terminal batch missing its files.
    pub fn batch_finish(
        &self,
        id: &str,
        status: &str,
        files: &[BatchFileRow],
        output_file_id: Option<&str>,
        error_file_id: Option<&str>,
        webhook: bool,
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        for f in files {
            tx.execute(
                "INSERT INTO batch_files (id, purpose, filename, bytes, created_ms, path, batch_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET bytes = excluded.bytes",
                params![
                    f.id,
                    f.purpose,
                    f.filename,
                    f.bytes,
                    f.created_ms,
                    f.path,
                    f.batch_id
                ],
            )?;
        }
        let col = stamp_col(status).unwrap_or("completed_ms");
        tx.execute(
            &format!(
                "UPDATE batches SET status = ?2, {col} = COALESCE({col}, ?3),
                   output_file_id = ?4, error_file_id = ?5,
                   webhook_state = CASE WHEN ?6 THEN 1 ELSE webhook_state END
                 WHERE id = ?1"
            ),
            params![id, status, now_ms(), output_file_id, error_file_id, webhook],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn batch_webhook_set(
        &self,
        id: &str,
        state: i64,
        attempts: i64,
        next_ms: i64,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE batches SET webhook_state = ?2, webhook_attempts = ?3, webhook_next_ms = ?4
             WHERE id = ?1",
            params![id, state, attempts, next_ms],
        )?;
        Ok(())
    }

    /// Drop a batch's rows (items included); the caller removes its bytes.
    pub fn batch_delete(&self, id: &str) -> Result<Vec<BatchFileRow>, StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let files: Vec<BatchFileRow> = {
            let mut stmt = tx.prepare(
                "SELECT id, purpose, filename, bytes, created_ms, path, batch_id
                 FROM batch_files WHERE batch_id = ?1",
            )?;
            stmt.query_map(params![id], file_row)?
                .collect::<Result<_, _>>()?
        };
        tx.execute("DELETE FROM batch_files WHERE batch_id = ?1", params![id])?;
        tx.execute("DELETE FROM batches WHERE id = ?1", params![id])?;
        tx.commit()?;
        drop(conn);
        // normally gone already (purged at finalize); chunked either way
        self.batch_items_purge(id)?;
        Ok(files)
    }

    /// Index rows for a batch's input, in one transaction per call - callers
    /// feed bounded chunks so a 100 000-line index never holds the store lock
    /// for the whole scan.
    pub fn batch_items_insert(
        &self,
        batch_id: &str,
        rows: &[(i64, String, i64, i64)],
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO batch_items (batch_id, idx, custom_id, off, len)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (idx, cid, off, len) in rows {
                stmt.execute(params![batch_id, idx, cid, off, len])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Delete a batch's item rows, 2 000 per transaction with a pause in
    /// between. One DELETE of a million rows would hold the store, and with it
    /// the Studio, for seconds. Chunks alone are not enough either: the store's
    /// mutex is not fair, so a loop that re-locks at once starves every other
    /// caller for the whole purge (measured: one Studio read waited 1.9 s at
    /// 1M). Returns how many went.
    pub fn batch_items_purge(&self, batch_id: &str) -> Result<usize, StoreError> {
        let mut total = 0;
        loop {
            let n = self.lock().execute(
                "DELETE FROM batch_items WHERE batch_id = ?1 AND idx IN (
                   SELECT idx FROM batch_items WHERE batch_id = ?1 ORDER BY idx LIMIT 2000)",
                params![batch_id],
            )?;
            total += n;
            if n == 0 {
                return Ok(total);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Finished batches whose item rows are still there: a purge a crash
    /// interrupted. Boot finishes the job.
    pub fn batch_items_leftover(&self) -> Result<Vec<String>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id FROM batches
             WHERE status IN ('completed', 'failed', 'expired', 'cancelled')
               AND EXISTS (SELECT 1 FROM batch_items i WHERE i.batch_id = batches.id)",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// The next pending items past `after_idx`, in input order.
    pub fn batch_items_pending(
        &self,
        batch_id: &str,
        after_idx: i64,
        limit: usize,
    ) -> Result<Vec<BatchItemRow>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            // pinned: on a fresh table the planner prefers the primary key, which
            // walks every finished row before the first pending one
            "SELECT idx, custom_id, off, len, attempts, strikes
             FROM batch_items INDEXED BY batch_items_pending
             WHERE batch_id = ?1 AND state = 0 AND idx > ?2 ORDER BY idx LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![batch_id, after_idx, limit as i64], |r| {
            Ok(BatchItemRow {
                idx: r.get(0)?,
                custom_id: r.get(1)?,
                off: r.get(2)?,
                len: r.get(3)?,
                attempts: r.get(4)?,
                strikes: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Apply one group commit atomically (see the module doc for why the
    /// lengths ride in the same transaction as the item states).
    pub fn batch_commit(&self, batch_id: &str, c: &BatchCommit) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "UPDATE batch_items SET state = ?3, attempts = ?4, strikes = ?5
                 WHERE batch_id = ?1 AND idx = ?2",
            )?;
            for u in &c.items {
                stmt.execute(params![batch_id, u.idx, u.state, u.attempts, u.strikes])?;
            }
        }
        tx.execute(
            "UPDATE batches SET out_len = ?2, err_len = ?3,
               succeeded = succeeded + ?4, errored = errored + ?5,
               canceled = canceled + ?6, expired = expired + ?7,
               input_tokens = input_tokens + ?8, output_tokens = output_tokens + ?9,
               cached_tokens = cached_tokens + ?10, reasoning_tokens = reasoning_tokens + ?11
             WHERE id = ?1",
            params![
                batch_id,
                c.out_len,
                c.err_len,
                c.succeeded,
                c.errored,
                c.canceled,
                c.expired,
                c.input_tokens,
                c.output_tokens,
                c.cached_tokens,
                c.reasoning_tokens
            ],
        )?;
        tx.commit()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UploadRow {
    pub id: String,
    pub filename: String,
    pub purpose: String,
    pub mime_type: String,
    pub bytes: i64,
    pub created_ms: i64,
    pub expires_ms: i64,
    /// pending | completed | cancelled | expired
    pub status: String,
    /// The part order the first completion fixed (JSON array of part ids).
    pub part_order: Option<String>,
    /// Parts appended to the assembly so far, and its committed length.
    pub assembled: i64,
    pub assembled_bytes: i64,
    pub file_id: Option<String>,
}

fn upload_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<UploadRow> {
    Ok(UploadRow {
        id: r.get(0)?,
        filename: r.get(1)?,
        purpose: r.get(2)?,
        mime_type: r.get(3)?,
        bytes: r.get(4)?,
        created_ms: r.get(5)?,
        expires_ms: r.get(6)?,
        status: r.get(7)?,
        part_order: r.get(8)?,
        assembled: r.get(9)?,
        assembled_bytes: r.get(10)?,
        file_id: r.get(11)?,
    })
}

const UPLOAD_COLS: &str = "id, filename, purpose, mime_type, bytes, created_ms, expires_ms, status,
    part_order, assembled, assembled_bytes, file_id";

impl Store {
    pub fn upload_insert(&self, u: &UploadRow) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO batch_uploads (id, filename, purpose, mime_type, bytes, created_ms,
               expires_ms, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                u.id,
                u.filename,
                u.purpose,
                u.mime_type,
                u.bytes,
                u.created_ms,
                u.expires_ms,
                u.status
            ],
        )?;
        Ok(())
    }

    pub fn upload_get(&self, id: &str) -> Result<Option<UploadRow>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                &format!("SELECT {UPLOAD_COLS} FROM batch_uploads WHERE id = ?1"),
                params![id],
                upload_row,
            )
            .optional()?)
    }

    pub fn upload_part_insert(
        &self,
        upload_id: &str,
        part_id: &str,
        bytes: i64,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO batch_upload_parts (upload_id, part_id, bytes, created_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![upload_id, part_id, bytes, now_ms()],
        )?;
        Ok(())
    }

    /// part id -> bytes, for one upload.
    pub fn upload_parts(&self, upload_id: &str) -> Result<Vec<(String, i64)>, StoreError> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT part_id, bytes FROM batch_upload_parts WHERE upload_id = ?1")?;
        let rows = stmt.query_map(params![upload_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Fix the part order (first completion) or confirm it matches.
    pub fn upload_set_order(&self, id: &str, order: &str) -> Result<bool, StoreError> {
        let conn = self.lock();
        conn.execute(
            "UPDATE batch_uploads SET part_order = ?2 WHERE id = ?1 AND part_order IS NULL",
            params![id, order],
        )?;
        let now: Option<String> = conn.query_row(
            "SELECT part_order FROM batch_uploads WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        Ok(now.as_deref() == Some(order))
    }

    /// One more part is in the assembly, durably (the caller synced it).
    pub fn upload_assembled(&self, id: &str, parts: i64, bytes: i64) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE batch_uploads SET assembled = ?2, assembled_bytes = ?3 WHERE id = ?1",
            params![id, parts, bytes],
        )?;
        Ok(())
    }

    /// The upload became a file: both rows in one transaction.
    pub fn upload_finish(&self, id: &str, file: &BatchFileRow) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO batch_files (id, purpose, filename, bytes, created_ms, path, batch_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL)
             ON CONFLICT(id) DO NOTHING",
            params![
                file.id,
                file.purpose,
                file.filename,
                file.bytes,
                file.created_ms,
                file.path
            ],
        )?;
        tx.execute(
            "UPDATE batch_uploads SET status = 'completed', file_id = ?2 WHERE id = ?1",
            params![id, file.id],
        )?;
        tx.execute(
            "DELETE FROM batch_upload_parts WHERE upload_id = ?1",
            params![id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// pending -> cancelled | expired (false when it was no longer pending).
    pub fn upload_close(&self, id: &str, status: &str) -> Result<bool, StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let n = tx.execute(
            "UPDATE batch_uploads SET status = ?2 WHERE id = ?1 AND status = 'pending'",
            params![id, status],
        )?;
        tx.execute(
            "DELETE FROM batch_upload_parts WHERE upload_id = ?1",
            params![id],
        )?;
        tx.commit()?;
        Ok(n > 0)
    }

    /// Pending uploads past their deadline.
    pub fn uploads_expired(&self, now: i64) -> Result<Vec<String>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT id FROM batch_uploads WHERE status = 'pending' AND expires_ms < ?1")?;
        let rows = stmt.query_map(params![now], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

#[cfg(test)]
impl Store {
    /// Move a batch's deadline (the expiry tests).
    pub fn batch_set_expires(&self, id: &str, ms: i64) {
        self.lock()
            .execute(
                "UPDATE batches SET expires_ms = ?2 WHERE id = ?1",
                params![id, ms],
            )
            .unwrap();
    }

    pub fn batch_items_count(&self, id: &str) -> i64 {
        self.lock()
            .query_row(
                "SELECT COUNT(*) FROM batch_items WHERE batch_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(s: &Store, sql: &str) -> String {
        let conn = s.lock();
        let mut stmt = conn.prepare(&format!("EXPLAIN QUERY PLAN {sql}")).unwrap();
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(3))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        rows.join(" | ")
    }

    #[test]
    fn the_hot_queries_use_the_partial_indexes() {
        let s = Store::open(&PathBuf::from(":memory:")).unwrap();
        let p = plan(
            &s,
            "SELECT idx, custom_id, off, len, attempts, strikes
             FROM batch_items INDEXED BY batch_items_pending
             WHERE batch_id = 'b' AND state = 0 AND idx > -1 ORDER BY idx LIMIT 10",
        );
        assert!(p.contains("batch_items_pending"), "{p}");
        let p = plan(
            &s,
            &format!("SELECT id FROM batches WHERE {OPEN} ORDER BY created_ms ASC, id ASC"),
        );
        assert!(p.contains("batches_open"), "{p}");
    }

    #[test]
    fn purge_goes_in_chunks_and_leftovers_are_found() {
        let s = Store::open(&PathBuf::from(":memory:")).unwrap();
        let b = BatchRow {
            id: "batch_x".into(),
            dialect: "openai".into(),
            status: "in_progress".into(),
            ..Default::default()
        };
        s.batch_insert(&b).unwrap();
        let rows: Vec<(i64, String, i64, i64)> = (0..12_345)
            .map(|i| (i, format!("c{i}"), i * 10, 9))
            .collect();
        for c in rows.chunks(2_000) {
            s.batch_items_insert("batch_x", c).unwrap();
        }
        assert!(s.batch_items_leftover().unwrap().is_empty(), "still open");
        s.batch_finish("batch_x", "completed", &[], None, None, false)
            .unwrap();
        assert_eq!(s.batch_items_leftover().unwrap(), ["batch_x"]);
        assert_eq!(s.batch_items_purge("batch_x").unwrap(), 12_345);
        assert!(s.batch_items_leftover().unwrap().is_empty());
    }
}
