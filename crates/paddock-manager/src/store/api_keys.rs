//! API keys: named keys for the network API. Only a hash of each key is stored;
//! the plaintext is shown once at creation.
use super::*;

fn hash_key(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    crate::registry::hex(&h.finalize())
}

impl Store {
    /// Create a named key. Returns `(public_record, plaintext_key)` - the
    /// plaintext is shown once and never stored (only its hash).
    pub fn create_api_key(&self, name: &str) -> Result<(Value, String), StoreError> {
        let id = Uuid::new_v4().to_string();
        let key = format!("pk-{}", Uuid::new_v4().simple());
        let prefix = key.chars().take(9).collect::<String>();
        let created = now_ms();
        self.lock().execute(
            "INSERT INTO api_keys (id, name, key_hash, prefix, created_at) VALUES (?1,?2,?3,?4,?5)",
            params![id, name, hash_key(&key), prefix, created],
        )?;
        let record = json!({ "id": id, "name": name, "prefix": prefix, "createdAt": created, "revoked": false });
        Ok((record, key))
    }

    pub fn list_api_keys(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, prefix, created_at, last_used_at, revoked
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
            }))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn revoke_api_key(&self, id: &str) -> Result<(), StoreError> {
        self.lock()
            .execute("UPDATE api_keys SET revoked = 1 WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// True if the key matches a non-revoked stored key; also stamps last_used.
    pub fn verify_api_key(&self, key: &str) -> Result<bool, StoreError> {
        let hash = hash_key(key);
        let conn = self.lock();
        let id: Option<String> = conn
            .query_row(
                "SELECT id FROM api_keys WHERE key_hash = ?1 AND revoked = 0",
                params![hash],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = id {
            conn.execute(
                "UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1",
                params![id, now_ms()],
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
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
