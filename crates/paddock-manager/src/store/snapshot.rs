//! The sanitized export's raw material: a consistent copy of the live DB.
use super::*;

impl Store {
    /// A CONSISTENT copy of the DB written to a fresh temp file (`VACUUM INTO`,
    /// so no WAL/locking surprises). The caller owns the file - sanitize it,
    /// read it, then delete it. Used by the sanitized export endpoint.
    pub fn snapshot_to_temp(&self) -> Result<PathBuf, StoreError> {
        // The unsanitized snapshot contains credentials. Random exclusive
        // creation and mode 0600 apply BEFORE SQLite starts writing.
        let tmp = tempfile::Builder::new()
            .prefix("paddock-export-")
            .suffix(".db")
            .tempfile()
            .map_err(|e| StoreError::Bad(e.to_string()))?
            .into_temp_path();
        let conn = self.lock();
        conn.execute("VACUUM INTO ?1", params![tmp.to_string_lossy()])?;
        tmp.keep().map_err(|e| StoreError::Bad(e.to_string()))
    }
}
