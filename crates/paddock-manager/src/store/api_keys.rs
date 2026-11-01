//! API keys: named keys for the network API. Only a hash of each key is stored;
//! the plaintext is shown once at creation.
//!
//! Every key carries a scope. `inference` keys open the manager's own `/v1`
//! surface (files and batches) and nothing else; `admin` keys open the
//! Studio's `/api` control plane too - spawning models, writing their
//! configs, minting keys. A serving key must never be able to reconfigure the
//! box, so `inference` is the default and `admin` is asked for by name.
use super::*;

pub(super) fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS api_keys (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            key_hash     TEXT NOT NULL,
            prefix       TEXT NOT NULL,
            created_at   INTEGER NOT NULL,
            last_used_at INTEGER,
            revoked      INTEGER NOT NULL DEFAULT 0,
            scope        TEXT NOT NULL DEFAULT 'inference'
        );",
    )?;
    // scope arrived after the table shipped. Keys minted before it were
    // all-powerful by accident, not by choice, so they land on `inference`:
    // the configured manager key is unaffected (it is admin by definition),
    // and an operator who wants another admin key mints one by name.
    let has_scope = conn
        .prepare("PRAGMA table_info(api_keys)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|c| c == "scope");
    if !has_scope {
        conn.execute(
            "ALTER TABLE api_keys ADD COLUMN scope TEXT NOT NULL DEFAULT 'inference'",
            [],
        )?;
    }
    Ok(())
}

/// What a key may open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyScope {
    /// The manager's `/v1` surface only.
    Inference,
    /// Everything, the Studio's `/api` control plane included.
    Admin,
}

impl KeyScope {
    pub fn as_str(self) -> &'static str {
        match self {
            KeyScope::Inference => "inference",
            KeyScope::Admin => "admin",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "inference" => Some(KeyScope::Inference),
            "admin" => Some(KeyScope::Admin),
            _ => None,
        }
    }
}

fn hash_key(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    crate::registry::hex(&h.finalize())
}

impl Store {
    /// Create a named key. Returns `(public_record, plaintext_key)` - the
    /// plaintext is shown once and never stored (only its hash).
    pub fn create_api_key(
        &self,
        name: &str,
        scope: KeyScope,
    ) -> Result<(Value, String), StoreError> {
        let id = Uuid::new_v4().to_string();
        let key = format!("pk-{}", Uuid::new_v4().simple());
        let prefix = key.chars().take(9).collect::<String>();
        let created = now_ms();
        self.lock().execute(
            "INSERT INTO api_keys (id, name, key_hash, prefix, created_at, scope)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, name, hash_key(&key), prefix, created, scope.as_str()],
        )?;
        let record = json!({
            "id": id,
            "name": name,
            "prefix": prefix,
            "createdAt": created,
            "revoked": false,
            "scope": scope.as_str(),
        });
        Ok((record, key))
    }

    pub fn list_api_keys(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, prefix, created_at, last_used_at, revoked, scope
             FROM api_keys ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "name": r.get::<_, String>(1)?,
                "prefix": r.get::<_, String>(2)?,
                "createdAt": r.get::<_, i64>(3)?,
                "lastUsedAt": r.get::<_, Option<i64>>(4)?,
                "revoked": r.get::<_, i64>(5)? != 0,
                "scope": r.get::<_, String>(6)?,
            }))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn revoke_api_key(&self, id: &str) -> Result<(), StoreError> {
        self.lock()
            .execute("UPDATE api_keys SET revoked = 1 WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// The scope of a non-revoked stored key, or None; also stamps last_used.
    /// A scope this build does not know reads as None - an unknown grant is
    /// no grant.
    pub fn verify_api_key(&self, key: &str) -> Result<Option<KeyScope>, StoreError> {
        let hash = hash_key(key);
        let conn = self.lock();
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT id, scope FROM api_keys WHERE key_hash = ?1 AND revoked = 0",
                params![hash],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((id, scope)) = row else {
            return Ok(None);
        };
        conn.execute(
            "UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1",
            params![id, now_ms()],
        )?;
        Ok(KeyScope::parse(&scope))
    }

    /// Number of non-revoked keys (used by the auth policy).
    pub fn active_key_count(&self) -> Result<i64, StoreError> {
        Ok(self
            .lock()
            .query_row("SELECT COUNT(*) FROM api_keys WHERE revoked = 0", [], |r| {
                r.get(0)
            })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (std::path::PathBuf, Store) {
        let dir = std::env::temp_dir().join(format!("paddock-keys-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let db = Store::open(&dir.join("paddock.db")).expect("open store");
        (dir, db)
    }

    #[test]
    fn a_key_reports_the_scope_it_was_minted_with() {
        let (dir, db) = temp_db();
        let (rec, serving) = db
            .create_api_key("serving", KeyScope::Inference)
            .expect("mint");
        assert_eq!(rec["scope"], "inference");
        let (_, admin) = db.create_api_key("ops", KeyScope::Admin).expect("mint");
        assert_eq!(
            db.verify_api_key(&serving).expect("verify"),
            Some(KeyScope::Inference)
        );
        assert_eq!(
            db.verify_api_key(&admin).expect("verify"),
            Some(KeyScope::Admin)
        );
        assert_eq!(db.verify_api_key("pk-not-a-key").expect("verify"), None);
        let id = rec["id"].as_str().expect("id").to_owned();
        db.revoke_api_key(&id).expect("revoke");
        assert_eq!(db.verify_api_key(&serving).expect("verify"), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A database from before scopes: its keys keep working, as inference
    /// keys, and the column arrives without touching anything else.
    #[test]
    fn keys_from_before_scopes_become_inference_keys() {
        let dir = std::env::temp_dir().join(format!("paddock-keys-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("paddock.db");
        let key = "pk-from-an-older-build";
        {
            let conn = Connection::open(&path).expect("open");
            conn.execute_batch(
                "CREATE TABLE api_keys (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL,
                    prefix TEXT NOT NULL, created_at INTEGER NOT NULL,
                    last_used_at INTEGER, revoked INTEGER NOT NULL DEFAULT 0
                );",
            )
            .expect("old schema");
            conn.execute(
                "INSERT INTO api_keys (id, name, key_hash, prefix, created_at)
                 VALUES ('old', 'old', ?1, 'pk-from-a', 1)",
                params![hash_key(key)],
            )
            .expect("old row");
        }
        let db = Store::open(&path).expect("migrate");
        assert_eq!(
            db.verify_api_key(key).expect("verify"),
            Some(KeyScope::Inference)
        );
        let listed = db.list_api_keys().expect("list");
        assert_eq!(listed[0]["scope"], "inference");
        drop(db);
        // and a second open is a no-op
        let db = Store::open(&path).expect("reopen");
        assert_eq!(db.list_api_keys().expect("list").len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
