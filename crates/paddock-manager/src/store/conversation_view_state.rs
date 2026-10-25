//! Small presentation checkpoints. They never rewrite a transcript or change
//! its history ordering, and cannot resurrect a deleted conversation.
use super::*;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS conversation_view_state (
            conversation_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            doc TEXT NOT NULL,
            PRIMARY KEY (conversation_id, scope)
        );",
    )?;
    Ok(())
}

fn key(id: &str, scope: &str) -> Result<(), StoreError> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        || !matches!(scope, "macos" | "web")
    {
        return Err(StoreError::Bad(
            "Invalid conversation view-state key".into(),
        ));
    }
    Ok(())
}

fn valid(doc: &Value) -> bool {
    if doc["version"] != 1 || doc.to_string().len() > 2048 {
        return false;
    }
    if let Some(reading) = doc.get("reading") {
        let number = |name: &str, min: f64, max: f64| {
            reading[name]
                .as_f64()
                .is_some_and(|n| n.is_finite() && n >= min && n <= max)
        };
        if !reading.is_null()
            && !(reading["following"].is_boolean()
                && reading["blockID"].as_str().is_some_and(|s| s.len() <= 256)
                && reading["index"].as_u64().is_some_and(|n| n <= 1_000_000)
                && number("offset", -4096.0, 1e9)
                && number("height", 0.0, 1e9)
                && number("width", 0.0, 1e6))
        {
            return false;
        }
    }
    if let Some(composer) = doc.get("composer")
        && !composer.is_null()
        && !(composer["location"].as_u64().is_some_and(|n| n <= 262_144)
            && composer["length"].as_u64().is_some_and(|n| n <= 262_144)
            && composer["digest"]
                .as_str()
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())))
    {
        return false;
    }
    true
}

impl Store {
    pub fn get_conversation_view_state(
        &self,
        id: &str,
        scope: &str,
    ) -> Result<Option<Value>, StoreError> {
        key(id, scope)?;
        let body: Option<String> = self
            .lock()
            .query_row(
                "SELECT doc FROM conversation_view_state WHERE conversation_id=?1 AND scope=?2",
                params![id, scope],
                |row| row.get(0),
            )
            .optional()?;
        body.map(|s| serde_json::from_str(&s).map_err(|e| StoreError::Bad(e.to_string())))
            .transpose()
    }

    pub fn put_conversation_view_state(
        &self,
        id: &str,
        scope: &str,
        doc: &Value,
    ) -> Result<bool, StoreError> {
        key(id, scope)?;
        if !valid(doc) {
            return Err(StoreError::Bad(
                "Invalid or oversized conversation view state".into(),
            ));
        }
        // One SQLite statement checks ownership and writes atomically with
        // respect to deletion, draft promotion and other connections.
        let count = self.lock().execute(
            "INSERT INTO conversation_view_state(conversation_id,scope,doc)
             SELECT ?1,?2,?3 WHERE EXISTS(SELECT 1 FROM conversations WHERE id=?1)
                OR EXISTS(SELECT 1 FROM conversation_drafts WHERE id=?1)
             ON CONFLICT(conversation_id,scope) DO UPDATE SET doc=excluded.doc",
            params![id, scope, doc.to_string()],
        )?;
        Ok(count > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn view() -> Value {
        json!({"version":1,"reading":{"following":false,"blockID":"answer","index":1,"offset":234.5,"height":800,"width":760},
            "composer":{"location":2,"length":3,"digest":"a".repeat(64)}})
    }
    #[test]
    fn view_state_survives_reopen_and_snapshot_without_changing_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.db");
        let db = Store::open(&path).unwrap();
        let doc = json!({"id":"chat","title":"Kept","messages":[],"updatedAt":1234});
        db.put_conversation(&doc).unwrap();
        let history = db.list_conversations().unwrap();
        assert!(
            db.put_conversation_view_state("chat", "macos", &view())
                .unwrap()
        );
        assert_eq!(history, db.list_conversations().unwrap());
        assert_eq!(db.get_conversation("chat").unwrap(), Some(doc));
        let snapshot = db.snapshot_to_temp().unwrap();
        drop(db);
        for file in [&path, &snapshot] {
            let reopened = Store::open(file).unwrap();
            assert_eq!(
                reopened
                    .get_conversation_view_state("chat", "macos")
                    .unwrap(),
                Some(view())
            );
            assert!(
                reopened
                    .get_conversation_view_state("chat", "web")
                    .unwrap()
                    .is_none()
            );
        }
        std::fs::remove_file(snapshot).unwrap();
    }
    #[test]
    fn deletion_and_empty_drafts_remove_state_and_late_writes_cannot_resurrect_it() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        assert!(
            !db.put_conversation_view_state("missing", "macos", &view())
                .unwrap()
        );
        let draft = json!({"text":"Draft","revision":"","attachments":[],"conversation":{"id":"chat","title":"Draft","model":"m","messages":[]}});
        let rev = db.put_conversation_draft("chat", &draft).unwrap();
        db.put_conversation_view_state("chat", "macos", &view())
            .unwrap();
        let mut empty = draft.clone();
        empty["text"] = json!("");
        empty["revision"] = rev["revision"].clone();
        db.put_conversation_draft("chat", &empty).unwrap();
        assert!(
            db.get_conversation_view_state("chat", "macos")
                .unwrap()
                .is_none()
        );
        db.put_conversation(&json!({"id":"chat","title":"Chat","messages":[]}))
            .unwrap();
        db.put_conversation_view_state("chat", "macos", &view())
            .unwrap();
        db.delete_conversation("chat").unwrap();
        assert!(
            !db.put_conversation_view_state("chat", "macos", &view())
                .unwrap()
        );
        assert!(
            db.get_conversation_view_state("chat", "macos")
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn malformed_state_is_rejected() {
        let db = Store::open(&PathBuf::from(":memory:")).unwrap();
        assert!(
            db.put_conversation_view_state("../x", "macos", &view())
                .is_err()
        );
        assert!(
            db.put_conversation_view_state("chat", "arbitrary", &view())
                .is_err()
        );
        for (key, value) in [
            ("offset", json!(-10000)),
            ("width", json!("wrong")),
            ("index", json!(-1)),
        ] {
            let mut doc = view();
            doc["reading"][key] = value;
            assert!(
                db.put_conversation_view_state("chat", "macos", &doc)
                    .is_err()
            );
        }
    }
}
