//! Masks history - the Masks page's earlier pictures (`/api/mask-history`),
//! kept the way tables and reads are: summaries for the side panel, the whole
//! record when one is opened. A record is a picture, its prompts with what
//! each found, and the camera snapshots kept while it was open; the pictures
//! ride in it once each, keyed by a hash of their data URL.
//!
//! The document is kept as the TEXT the Studio sent and handed back byte for
//! byte (the read history's rule: a round trip through `Value` would reorder
//! its keys). Writes are revision-checked like the table history's: a save
//! names the revision it was made on, and one made on a stale revision is a
//! conflict, not an overwrite; the same text sent twice (a lost reply) is
//! answered with the row it already made.
use super::*;

/// A record holds its picture (up to 24 MP, as a data URL) and the
/// snapshots kept with it.
const LIMIT: usize = 64 * 1024 * 1024;
const MAX_LAYERS: usize = 64;
const MAX_SNAPSHOTS: usize = 48;

#[cfg(test)]
#[path = "mask_history_tests.rs"]
mod tests;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS mask_history (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            runs INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            revision TEXT NOT NULL,
            doc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS mask_history_recent ON mask_history(updated_at DESC, id);",
    )?;
    Ok(())
}

fn hash(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// `runs` is how many prompts have found something - the count the side
// panel shows, as a read's runs are
fn summary(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(
        json!({"id":r.get::<_,String>(0)?,"title":r.get::<_,String>(1)?,
        "model":r.get::<_,String>(2)?,"runs":r.get::<_,i64>(3)?,
        "createdAt":r.get::<_,i64>(4)?,"updatedAt":r.get::<_,i64>(5)?,
        "revision":r.get::<_,String>(6)?}),
    )
}

const COLS: &str = "id,title,model,runs,created_at,updated_at,revision";

/// The shape the list and a reopen rely on; the prompts' results are the
/// endpoint's own answers and are kept as they came.
fn validate(id: &str, doc: &Value) -> Result<usize, StoreError> {
    let bad = || StoreError::Bad("Invalid masks document".into());
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
    let pictures = doc["pictures"].as_object().ok_or_else(bad)?;
    for (key, url) in pictures {
        let url = url.as_str().ok_or_else(bad)?;
        if !url.starts_with("data:image/") || key != &hash(url) {
            return Err(bad());
        }
    }
    let size = |v: &Value| v.as_u64().is_some_and(|n| (1..=1 << 16).contains(&n));
    let picture = &doc["picture"];
    if !picture.is_null()
        && (!picture["ref"]
            .as_str()
            .is_some_and(|k| pictures.contains_key(k))
            || !picture["name"].as_str().is_some_and(|s| s.len() <= 1024)
            || !size(&picture["width"])
            || !size(&picture["height"]))
    {
        return Err(bad());
    }
    let layers = doc["layers"].as_array().ok_or_else(bad)?;
    if layers.is_empty() || layers.len() > MAX_LAYERS {
        return Err(bad());
    }
    let mut found = 0;
    for l in layers {
        if !matches!(l["kind"].as_str(), Some("concept" | "object"))
            || !l["text"].as_str().is_some_and(|s| s.len() <= 1024)
            || !l["boxes"].is_array()
            || !l["points"].is_array()
            || !l["hidden"].is_array()
            || !l["threshold"]
                .as_f64()
                .is_some_and(|t| (0.0..=1.0).contains(&t))
            || !(l["result"].is_null() || l["result"].is_object())
        {
            return Err(bad());
        }
        found += l["result"].is_object() as usize;
    }
    let shots = doc["snapshots"].as_array().ok_or_else(bad)?;
    if shots.len() > MAX_SNAPSHOTS {
        return Err(StoreError::Bad(format!(
            "A picture keeps at most {MAX_SNAPSHOTS} snapshots. Remove some first."
        )));
    }
    let mut ids = std::collections::HashSet::new();
    for s in shots {
        let sid = s["id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 128)
            .ok_or_else(bad)?;
        if !ids.insert(sid)
            || !s["ref"].as_str().is_some_and(|k| pictures.contains_key(k))
            || !size(&s["width"])
            || !size(&s["height"])
            || !s["at"].as_i64().is_some_and(|n| n >= 0)
            || !s["concepts"].is_array()
            || !s["objects"].is_array()
        {
            return Err(bad());
        }
    }
    Ok(found)
}

impl Store {
    pub fn list_mask_history(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {COLS} FROM mask_history ORDER BY updated_at DESC,id"
        ))?;
        Ok(stmt
            .query_map([], summary)?
            .collect::<Result<Vec<_>, _>>()?)
    }
    /// The record as it was saved, with its revision.
    pub fn get_mask_session(&self, id: &str) -> Result<Option<Value>, StoreError> {
        Ok(self
            .lock()
            .query_row(
                "SELECT doc,revision FROM mask_history WHERE id=?1",
                [id],
                |r| Ok(json!({"doc":r.get::<_,String>(0)?,"revision":r.get::<_,String>(1)?})),
            )
            .optional()?)
    }
    /// Save a whole record made on revision `expected` ("" for a new one).
    pub fn put_mask_session(
        &self,
        id: &str,
        text: &str,
        expected: &str,
    ) -> Result<Value, StoreError> {
        if text.len() > LIMIT {
            return Err(StoreError::Bad(
                "This picture's record is over 64 MiB. Remove some snapshots, or start a new picture; nothing on screen was lost."
                    .into(),
            ));
        }
        let doc: Value =
            serde_json::from_str(text).map_err(|_| StoreError::Bad("Invalid masks JSON".into()))?;
        let found = validate(id, &doc)?;
        let revision = hash(text);
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let old: Option<(String, i64)> = tx
            .query_row(
                "SELECT revision,created_at FROM mask_history WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if old.as_ref().map(|v| v.0.as_str()).unwrap_or("") != expected
            && old.as_ref().map(|v| v.0.as_str()) != Some(&revision)
        {
            return Err(StoreError::Conflict("This picture changed or was deleted in another window. What is on screen is kept; reopen it before saving.".into()));
        }
        if old
            .as_ref()
            .is_some_and(|v| Some(v.1) != doc["createdAt"].as_i64())
        {
            return Err(StoreError::Conflict(
                "A saved picture keeps its creation time. Reopen it before saving.".into(),
            ));
        }
        tx.execute("INSERT INTO mask_history(id,title,model,runs,created_at,updated_at,revision,doc) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET title=?2,model=?3,runs=?4,updated_at=?6,revision=?7,doc=?8",
            params![id, doc["title"].as_str(), doc["model"].as_str(), found as i64, doc["createdAt"].as_i64(), doc["updatedAt"].as_i64(), revision, text])?;
        let row = tx.query_row(
            &format!("SELECT {COLS} FROM mask_history WHERE id=?1"),
            [id],
            summary,
        )?;
        tx.commit()?;
        Ok(row)
    }
    pub fn delete_mask_session(&self, id: &str, expected: &str) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let current: Option<String> = tx
            .query_row("SELECT revision FROM mask_history WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .optional()?;
        if current.as_deref() != Some(expected) {
            return Err(StoreError::Conflict(
                "This picture changed or was deleted. Reload the list before deleting.".into(),
            ));
        }
        tx.execute("DELETE FROM mask_history WHERE id=?1", [id])?;
        tx.commit()?;
        Ok(())
    }
}
