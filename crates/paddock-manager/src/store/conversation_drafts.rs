//! Small, revision-checked composer records. No transcript copies or file bytes.
//! This is part of the same SQLite snapshot/backup as conversations and uploads.
use super::*;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS conversation_drafts (
            id TEXT PRIMARY KEY,
            revision TEXT NOT NULL,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            doc TEXT NOT NULL
        );",
    )?;
    Ok(())
}

pub(super) fn check_revision(
    conn: &Connection,
    id: &str,
    expected: &str,
) -> Result<(), StoreError> {
    let actual: Option<String> = conn
        .query_row(
            "SELECT revision FROM conversation_drafts WHERE id=?1",
            [id],
            |r| r.get(0),
        )
        .optional()?;
    if actual.as_deref().unwrap_or("") != expected {
        return Err(StoreError::Conflict(
            "This draft changed in another window. Your text has been kept; reopen the chat before replacing it.".into(),
        ));
    }
    Ok(())
}

impl Store {
    pub fn list_conversation_drafts(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id,title,model,updated_at FROM conversation_drafts ORDER BY updated_at DESC,id",
        )?;
        Ok(stmt
            .query_map([], |r| {
                Ok(json!({
                    "id":r.get::<_, String>(0)?, "title":r.get::<_, String>(1)?,
                    "model":r.get::<_, String>(2)?, "updatedAt":r.get::<_, i64>(3)?,
                    "composerDraft":true
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?)
    }

    pub fn get_conversation_draft(&self, id: &str) -> Result<Option<Value>, StoreError> {
        let conn = self.lock();
        let body: Option<String> = conn
            .query_row(
                "SELECT doc FROM conversation_drafts WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        body.map(|body| serde_json::from_str(&body).map_err(|e| StoreError::Bad(e.to_string())))
            .transpose()
    }

    pub fn put_conversation_draft(&self, id: &str, doc: &Value) -> Result<Value, StoreError> {
        let bad = || StoreError::Bad("Invalid or oversized composer draft".into());
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(bad());
        }
        let text = doc["text"]
            .as_str()
            .filter(|s| s.len() <= 128 * 1024)
            .ok_or_else(bad)?;
        let expected = doc["revision"]
            .as_str()
            .filter(|s| s.len() <= 128)
            .ok_or_else(bad)?;
        let attachments = doc["attachments"]
            .as_array()
            .filter(|a| a.len() <= 32)
            .ok_or_else(bad)?;
        let seed = &doc["conversation"];
        // Only the unsent conversation's configuration may be carried here.
        // An existing transcript is always loaded from its own authoritative row.
        if seed["id"] != id
            || seed["messages"].as_array().is_none_or(|m| !m.is_empty())
            || doc.to_string().len() > 8 * 1024 * 1024
        {
            return Err(bad());
        }
        let title = seed["title"]
            .as_str()
            .filter(|s| s.len() <= 512)
            .ok_or_else(bad)?;
        let model = seed["model"]
            .as_str()
            .filter(|s| s.len() <= 4096)
            .ok_or_else(bad)?;
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        check_revision(&tx, id, expected)?;
        let mut ids = std::collections::HashSet::new();
        for attachment in attachments {
            let aid = attachment["display"]["id"].as_str().ok_or_else(bad)?;
            if attachment["part"]["attachmentId"] != aid || !ids.insert(aid) {
                return Err(bad());
            }
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM attachments WHERE id=?1)",
                [aid],
                |r| r.get(0),
            )?;
            if !exists {
                return Err(StoreError::Bad(
                    "An attachment is not stored yet. Your draft has been kept.".into(),
                ));
            }
        }
        if text.is_empty() && attachments.is_empty() {
            tx.execute("DELETE FROM conversation_drafts WHERE id=?1", [id])?;
            tx.execute("DELETE FROM conversation_view_state WHERE conversation_id=?1 AND NOT EXISTS(SELECT 1 FROM conversations WHERE id=?1)", [id])?;
            tx.commit()?;
            return Ok(json!({"revision":""}));
        }
        let revision = Uuid::new_v4().to_string();
        let saved =
            json!({"revision":revision,"text":text,"attachments":attachments,"conversation":seed});
        tx.execute("INSERT INTO conversation_drafts(id,revision,title,model,updated_at,doc) VALUES(?1,?2,?3,?4,?5,?6)
            ON CONFLICT(id) DO UPDATE SET revision=?2,title=?3,model=?4,updated_at=?5,doc=?6",
            params![id,revision,title,model,now_ms(),saved.to_string()])?;
        tx.commit()?;
        Ok(json!({"revision":revision}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draft(text: &str, revision: &str) -> Value {
        json!({"text":text,"revision":revision,"attachments":[],
            "conversation":{"id":"chat","title":"Draft","model":"model","messages":[]}})
    }
    #[test]
    fn drafts_are_separate_and_consumed_atomically() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let rev = db
            .put_conversation_draft("chat", &draft("Hello", ""))
            .unwrap();
        assert!(db.list_conversations().unwrap().is_empty());
        assert_eq!(db.list_conversation_drafts().unwrap().len(), 1);
        let doc = json!({"id":"chat","title":"Reply","messages":[{"role":"user"}]});
        assert!(matches!(
            db.put_conversation_consuming_draft(&doc, Some("stale")),
            Err(StoreError::Conflict(_))
        ));
        assert!(db.get_conversation("chat").unwrap().is_none());
        assert!(db.get_conversation_draft("chat").unwrap().is_some());
        db.put_conversation_consuming_draft(&doc, rev["revision"].as_str())
            .unwrap();
        assert_eq!(db.get_conversation("chat").unwrap(), Some(doc));
        assert!(db.get_conversation_draft("chat").unwrap().is_none());
    }
    #[test]
    fn stale_writes_empty_clear_and_delete_preserve_conversations() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        let doc = json!({"id":"chat","title":"Original","messages":[{"role":"assistant"}]});
        db.put_conversation(&doc).unwrap();
        let rev = db
            .put_conversation_draft("chat", &draft("First", ""))
            .unwrap();
        assert!(
            db.put_conversation_draft("chat", &draft("Stale", ""))
                .is_err()
        );
        db.put_conversation_draft("chat", &draft("", rev["revision"].as_str().unwrap()))
            .unwrap();
        assert!(db.list_conversation_drafts().unwrap().is_empty());
        assert_eq!(db.get_conversation("chat").unwrap(), Some(doc));
        db.put_conversation_draft("chat", &draft("Again", ""))
            .unwrap();
        db.delete_conversation("chat").unwrap();
        assert!(db.get_conversation_draft("chat").unwrap().is_none());
    }
    #[test]
    fn bounds_and_missing_uploads_are_rejected() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        assert!(
            db.put_conversation_draft("../bad", &draft("text", ""))
                .is_err()
        );
        assert!(
            db.put_conversation_draft("chat", &draft(&"x".repeat(128 * 1024 + 1), ""))
                .is_err()
        );
        let mut value = draft("file", "");
        value["attachments"] = json!([{"display":{"id":"file"},"part":{"attachmentId":"file"}}]);
        assert!(db.put_conversation_draft("chat", &value).is_err());
        db.put_attachment(
            "file",
            None,
            "application/pdf",
            "file.pdf",
            None,
            None,
            b"pdf",
        )
        .unwrap();
        db.put_conversation_draft("chat", &value).unwrap();
        assert_eq!(
            db.get_conversation_draft("chat").unwrap().unwrap()["attachments"],
            value["attachments"]
        );
    }

    #[test]
    fn drafts_survive_reopen_snapshot_and_reject_another_connections_stale_write() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("drafts.db");
        let first = Store::open(&path).unwrap();
        let revision = first
            .put_conversation_draft("chat", &draft("Durable", ""))
            .unwrap();
        let second = Store::open(&path).unwrap();
        assert_eq!(
            second.get_conversation_draft("chat").unwrap().unwrap()["text"],
            "Durable"
        );
        second
            .put_conversation_draft(
                "chat",
                &draft("Newer", revision["revision"].as_str().unwrap()),
            )
            .unwrap();
        assert!(matches!(
            first.put_conversation_draft(
                "chat",
                &draft("Stale", revision["revision"].as_str().unwrap())
            ),
            Err(StoreError::Conflict(_))
        ));
        let snapshot_path = first.snapshot_to_temp().unwrap();
        let snapshot = Store::open(&snapshot_path).unwrap();
        assert_eq!(
            snapshot.get_conversation_draft("chat").unwrap().unwrap()["text"],
            "Newer"
        );
        drop(snapshot);
        std::fs::remove_file(snapshot_path).unwrap();
    }
}
