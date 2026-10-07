//! Settings: the key/value preferences the Studio and the Manager share, and
//! the `secret.` rows beside them that hold credentials (a Hugging Face
//! token) - never listed, never patched, never exported.
use super::*;

/// Settings rows that hold credentials: kept out of `all_settings`, refused by
/// `patch_settings`, deleted from every export.
pub(crate) const SECRET_PREFIX: &str = "secret.";
const HF_TOKEN_KEY: &str = "secret.huggingface_token";

impl Store {
    /// The Hugging Face token saved in Manager > Settings (gated downloads,
    /// `registry/gated.rs`), resolved out of the Keychain where it lives there.
    pub fn huggingface_token(&self) -> Result<Option<String>, StoreError> {
        let stored: Option<String> = {
            let conn = self.lock();
            let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
            let mut rows = stmt.query(params![HF_TOKEN_KEY])?;
            match rows.next()? {
                Some(r) => {
                    let v: String = r.get(0)?;
                    serde_json::from_str::<Value>(&v)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                }
                None => None,
            }
        };
        match stored.filter(|s| !s.is_empty()) {
            Some(s) => crate::credentials::resolve(s)
                .map(Some)
                .map_err(StoreError::Bad),
            None => Ok(None),
        }
    }

    /// Save (`Some`) or forget (`None`) the Hugging Face token.
    pub fn set_huggingface_token(&self, token: Option<&str>) -> Result<(), StoreError> {
        match token {
            Some(t) => {
                let stored = crate::credentials::protect(t).map_err(StoreError::Bad)?;
                self.set_setting(HF_TOKEN_KEY, &Value::String(stored))
            }
            None => {
                self.lock()
                    .execute("DELETE FROM settings WHERE key = ?1", params![HF_TOKEN_KEY])?;
                Ok(())
            }
        }
    }

    pub fn all_settings(&self) -> Result<Value, StoreError> {
        let conn = self.lock();
        // `secret.` rows are credentials with their own endpoints; the
        // generic settings surface never shows them
        let mut stmt =
            conn.prepare("SELECT key, value FROM settings WHERE key NOT LIKE 'secret.%'")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        let mut map = serde_json::Map::new();
        for row in rows {
            let (k, v) = row?;
            map.insert(k, serde_json::from_str(&v).unwrap_or(Value::Null));
        }
        Ok(Value::Object(map))
    }

    pub fn set_setting(&self, key: &str, value: &Value) -> Result<(), StoreError> {
        let v = serde_json::to_string(value).map_err(|e| StoreError::Bad(e.to_string()))?;
        self.lock().execute(
            "INSERT INTO settings (key, value) VALUES (?1,?2)
             ON CONFLICT(key) DO UPDATE SET value=?2",
            params![key, v],
        )?;
        Ok(())
    }

    /// Commit a settings patch as one unit. Imports never overwrite an existing
    /// key (including a null tombstone), even with simultaneous Studio clients.
    pub fn patch_settings(&self, patch: &Value, import: bool) -> Result<(), StoreError> {
        let map = patch
            .as_object()
            .ok_or_else(|| StoreError::Bad("settings must be an object".into()))?;
        if map.len() > 128 || patch.to_string().len() > 1024 * 1024 {
            return Err(StoreError::Bad("settings patch is too large".into()));
        }
        for key in map.keys() {
            if key.is_empty() || key.len() > 160 {
                return Err(StoreError::Bad("invalid setting key".into()));
            }
            if key.starts_with(SECRET_PREFIX) {
                return Err(StoreError::Bad(
                    "credentials have their own endpoints, not the settings patch".into(),
                ));
            }
            if import && !key.starts_with("studio.") && key != "readsPanelOpen" {
                return Err(StoreError::Bad(
                    "only Studio preferences can be imported".into(),
                ));
            }
        }
        let mut conn = self.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        for (key, value) in map {
            tx.execute(
                if import {
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?1, ?2)"
                } else {
                    "INSERT INTO settings (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                },
                params![key, value.to_string()],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}
