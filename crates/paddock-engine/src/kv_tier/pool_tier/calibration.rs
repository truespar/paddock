//! The cost model's calibration across restarts (see `cost::Calibration`):
//! saved beside the T2 store it prices, keyed by the device it was measured
//! on, loaded when the tier arms.

use std::time::{Duration, Instant};

use super::super::cost::Calibration;
use super::*;

const FILE: &str = "cost-model.txt";

/// Rates drift slowly once measured; a changed calibration is written at
/// most this often (the first one at once, so a short run still leaves one).
const SAVE_EVERY: Duration = Duration::from_secs(30);

impl<T: XferSink> PoolTier<T> {
    /// Seed the cost model from the last run's calibration on this device,
    /// when one was saved beside the T2 store. Another device, an older
    /// format or a torn file leaves the generic seeds in place.
    pub(super) fn load_calibration(&mut self) {
        let Some((dir, device)) = self.transport.calibration_home() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(dir.join(FILE)) else {
            return;
        };
        match Calibration::decode(&text, &device) {
            Some(c) => {
                self.cost.adopt_calibration(c);
                tracing::info!(
                    prefill_tok_s = (c.prefill_tpus * 1e6).round(),
                    disk_gb_s = c.restore_nvme_bpus / 1e3,
                    "KV tier: cost model starts from this device's last calibration"
                );
            }
            None => tracing::debug!(
                "KV tier: saved calibration is for another device or format - generic seeds"
            ),
        }
    }

    /// Persist the calibration once this process has measured a prefill:
    /// at once the first time, then when it changed and `SAVE_EVERY` passed.
    /// Write-then-rename, so a reader never sees half a file.
    pub(super) fn save_calibration(&mut self) {
        let Some(c) = self.cost.calibration() else {
            return;
        };
        if self
            .cal_saved
            .is_some_and(|(at, prev)| prev == c || at.elapsed() < SAVE_EVERY)
        {
            return;
        }
        self.cal_saved = Some((Instant::now(), c));
        let Some((dir, device)) = self.transport.calibration_home() else {
            return;
        };
        let tmp = dir.join("cost-model.tmp");
        if let Err(e) = std::fs::write(&tmp, c.encode(&device))
            .and_then(|()| std::fs::rename(&tmp, dir.join(FILE)))
        {
            tracing::debug!(err = %e, "KV tier: calibration not saved");
        }
    }
}
