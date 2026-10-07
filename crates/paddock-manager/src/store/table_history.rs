//! Shared web/native table sessions. Lists never load datasets or predictions.
//! Complete documents use revision-checked, idempotent writes. Dataset snapshots
//! are content-addressed; runs keep their original inputs and full response.
use super::*;

const LIMIT: usize = 64 * 1024 * 1024;

#[cfg(test)]
#[path = "table_history_tests.rs"]
mod tests;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS table_history (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            runs INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            revision TEXT NOT NULL,
            doc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS table_history_recent ON table_history(updated_at DESC, id);",
    )?;
    Ok(())
}

fn hash(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn summary(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(
        json!({"id":r.get::<_,String>(0)?,"title":r.get::<_,String>(1)?,
        "model":r.get::<_,String>(2)?,"runs":r.get::<_,i64>(3)?,
        "createdAt":r.get::<_,i64>(4)?,"updatedAt":r.get::<_,i64>(5)?,
        "revision":r.get::<_,String>(6)?}),
    )
}

const COLS: &str = "id,title,model,runs,created_at,updated_at,revision";

fn validate(id: &str, doc: &Value) -> Result<(), StoreError> {
    let bad = || StoreError::Bad("Invalid table session document".into());
    if id.is_empty()
        || id.len() > 128
        || doc["id"].as_str() != Some(id)
        || doc["version"] != 1
        || !doc["title"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty() && s.len() <= 512)
        || !doc["model"].as_str().is_some_and(|s| s.len() <= 512)
        || !doc["createdAt"].as_i64().is_some_and(|n| n >= 0)
        || !doc["updatedAt"].as_i64().is_some_and(|n| n >= 0)
    {
        return Err(bad());
    }
    let datasets = doc["datasets"].as_object().ok_or_else(bad)?;
    for (key, value) in datasets {
        let source = value.as_str().ok_or_else(bad)?;
        if source.len() > 8 * 1024 * 1024 || key != &hash(source) {
            return Err(bad());
        }
    }
    let input = |v: &Value| -> Result<(), StoreError> {
        if !v["dataset"]
            .as_str()
            .is_some_and(|k| datasets.contains_key(k))
            || !v["fileName"].as_str().is_some_and(|s| s.len() <= 1024)
            || !v["model"].as_str().is_some_and(|s| s.len() <= 512)
            || !v["port"].as_u64().is_some_and(|p| p <= 65535)
            || !v["estimators"]
                .as_i64()
                .is_some_and(|n| (1..=16).contains(&n))
            || !v["seed"].as_u64().is_some_and(|n| n <= u32::MAX as u64)
        {
            return Err(bad());
        }
        if !v["spec"].is_null() {
            let s = &v["spec"];
            let types = s["types"].as_array().ok_or_else(bad)?;
            let used = s["use"].as_array().ok_or_else(bad)?;
            if types.is_empty()
                || types.len() > 501
                || used.len() != types.len()
                || !s["target"].as_u64().is_some_and(|n| n < types.len() as u64)
                || !used.iter().all(Value::is_boolean)
                || !types
                    .iter()
                    .all(|t| matches!(t.as_str(), Some("numerical" | "categorical")))
            {
                return Err(bad());
            }
        }
        Ok(())
    };
    input(&doc["draft"])?;
    let runs = doc["runs"].as_array().ok_or_else(bad)?;
    if runs.len() > 100 {
        return Err(StoreError::Bad(
            "Start a new table after 100 runs. Existing runs are kept.".into(),
        ));
    }
    let mut ids = std::collections::HashSet::new();
    for run in runs {
        let id = run["id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 128)
            .ok_or_else(bad)?;
        if !ids.insert(id)
            || !run["response"].is_object()
            || !matches!(run["task"].as_str(), Some("classification" | "regression"))
            || !run["at"].as_i64().is_some_and(|n| n >= 0)
            || !run["ms"].as_f64().is_some_and(|n| n.is_finite() && n >= 0.)
        {
            return Err(bad());
        }
        input(&run["input"])?;
    }
    Ok(())
}

impl Store {
    pub fn list_table_history(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {COLS} FROM table_history ORDER BY updated_at DESC,id"
        ))?;
        Ok(stmt
            .query_map([], summary)?
            .collect::<Result<Vec<_>, _>>()?)
    }
    pub fn get_table_session(&self, id: &str) -> Result<Option<Value>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                "SELECT doc,revision FROM table_history WHERE id=?1",
                [id],
                |r| Ok(json!({"doc":r.get::<_,String>(0)?,"revision":r.get::<_,String>(1)?})),
            )
            .optional()?)
    }
    pub fn put_table_session(
        &self,
        id: &str,
        text: &str,
        expected: &str,
    ) -> Result<Value, StoreError> {
        if text.len() > LIMIT {
            return Err(StoreError::Bad(
                "This table session exceeds 64 MiB. Start a new table; existing results are kept."
                    .into(),
            ));
        }
        let doc: Value = serde_json::from_str(text)
            .map_err(|_| StoreError::Bad("Invalid table session JSON".into()))?;
        validate(id, &doc)?;
        let revision = hash(text);
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let old: Option<(String, String)> = tx
            .query_row(
                "SELECT revision,doc FROM table_history WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if old.as_ref().map(|v| v.0.as_str()).unwrap_or("") != expected
            && old.as_ref().map(|v| v.0.as_str()) != Some(&revision)
        {
            return Err(StoreError::Conflict("This table changed or was deleted in another window. Your draft and results are kept; reopen it before saving.".into()));
        }
        if let Some((_, previous)) = &old {
            let previous: Value = serde_json::from_str(previous)
                .map_err(|_| StoreError::Bad("Invalid stored table".into()))?;
            let before = previous["runs"]
                .as_array()
                .ok_or_else(|| StoreError::Bad("Invalid stored table runs".into()))?;
            let after = doc["runs"].as_array().expect("validated table runs");
            if !after.starts_with(before) || doc["createdAt"] != previous["createdAt"] {
                return Err(StoreError::Conflict(
                    "Saved table runs are immutable. Reopen the table before saving.".into(),
                ));
            }
            for run in before {
                let key = run["input"]["dataset"]
                    .as_str()
                    .ok_or_else(|| StoreError::Bad("Invalid stored dataset reference".into()))?;
                if doc["datasets"][key] != previous["datasets"][key] {
                    return Err(StoreError::Bad(
                        "A saved run's dataset cannot change".into(),
                    ));
                }
            }
        }
        tx.execute("INSERT INTO table_history(id,title,model,runs,created_at,updated_at,revision,doc) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET title=?2,model=?3,runs=?4,updated_at=?6,revision=?7,doc=?8",
            params![id,doc["title"].as_str(),doc["model"].as_str(),doc["runs"].as_array().expect("validated table runs").len() as i64,doc["createdAt"].as_i64(),doc["updatedAt"].as_i64(),revision,text])?;
        let row = tx.query_row(
            &format!("SELECT {COLS} FROM table_history WHERE id=?1"),
            [id],
            summary,
        )?;
        tx.commit()?;
        Ok(row)
    }
    pub fn delete_table_session(&self, id: &str, expected: &str) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let current: Option<String> = tx
            .query_row(
                "SELECT revision FROM table_history WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        if current.as_deref() != Some(expected) {
            return Err(StoreError::Conflict(
                "This table changed or was deleted. Reload the history before deleting.".into(),
            ));
        }
        tx.execute("DELETE FROM table_history WHERE id=?1", [id])?;
        tx.commit()?;
        Ok(())
    }
}
