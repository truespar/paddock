//! CUDA device backend - the first execution target of the engine.
//!
//! cudarc with dynamic loading: binaries start on machines without CUDA and
//! report the absence as data, not a crash. P1 scope: device probe + health;
//! memory transfer and kernel launch land with the pack ABI v1.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::backend::{Backend, BackendInfo};

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("no CUDA driver or device available: {0}")]
    Unavailable(String),
    #[error("CUDA driver call failed: {0}")]
    Driver(String),
}

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    info: BackendInfo,
}

/// What a box without the NVIDIA driver is told. The driver is the one thing
/// paddock needs installed, so name it and the floor.
pub const NO_DRIVER: &str = "the CUDA driver library (libcuda.so / nvcuda.dll) is not installed; paddock needs the NVIDIA display driver, 580 or newer";

/// Is the CUDA driver library loadable at all? cudarc opens it lazily on the
/// first driver call and PANICS if it is missing, which is the one outcome the
/// dynamic-loading choice exists to avoid. Every first touch of the driver in
/// this crate asks here before it calls, so a machine without the driver gets
/// an error it can report instead of a crash.
///
/// Costs a dlopen, so it belongs at entry points, not on hot paths.
pub fn driver_present() -> bool {
    // SAFETY: only tries to open the shared library and drops the handle; it
    // calls no driver entry point and touches no state of ours.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Resolve a `gpu` config selector to a CUDA device ordinal - natively, with
/// no environment tricks (`CUDA_VISIBLE_DEVICES` is a launcher-era mechanism;
/// a config-file selection resolves against the driver directly). Accepts a
/// plain ordinal ("1") or a device UUID ("GPU-d56cd6c9-...", the NVML /
/// nvidia-smi spelling; case-insensitive, a unique prefix is enough) and
/// matches it against cuDeviceGetUuid across all devices.
pub fn resolve_device(selector: &str) -> Result<usize, CudaError> {
    let s = selector.trim();
    if s.is_empty() {
        return Ok(0);
    }
    if let Ok(n) = s.parse::<usize>() {
        return Ok(n);
    }
    let want = s
        .strip_prefix("GPU-")
        .unwrap_or(s)
        .replace('-', "")
        .to_lowercase();
    if want.is_empty() || !want.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(CudaError::Driver(format!(
            "gpu selector {selector:?} is neither a device ordinal nor a GPU-<uuid>"
        )));
    }
    if !driver_present() {
        return Err(CudaError::Unavailable(NO_DRIVER.into()));
    }
    cudarc::driver::result::init().map_err(|e| CudaError::Unavailable(e.to_string()))?;
    let count = cudarc::driver::result::device::get_count()
        .map_err(|e| CudaError::Unavailable(e.to_string()))? as usize;
    let mut hits = Vec::new();
    let mut seen = Vec::new();
    for ord in 0..count {
        let dev = cudarc::driver::result::device::get(ord as i32)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let uuid = cudarc::driver::result::device::get_uuid(dev)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let hex: String = uuid
            .bytes
            .iter()
            // c_char is i8 on x86 and u8 on aarch64: the byte, either way
            .map(|b| format!("{:02x}", b.to_ne_bytes()[0]))
            .collect();
        if hex.starts_with(&want) {
            hits.push(ord);
        }
        // 8-4-4-4-12, the spelling nvidia-smi prints - for the error message
        seen.push(format!(
            "GPU-{}-{}-{}-{}-{}",
            &hex[0..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        ));
    }
    match hits.as_slice() {
        [one] => Ok(*one),
        [] => Err(CudaError::Driver(format!(
            "no CUDA device matches {selector:?} (devices here: {})",
            if seen.is_empty() {
                "none".to_string()
            } else {
                seen.join(", ")
            }
        ))),
        _ => Err(CudaError::Driver(format!(
            "gpu selector {selector:?} matches {} devices - give more of the UUID",
            hits.len()
        ))),
    }
}

impl CudaBackend {
    /// Probe device `ordinal`. Failure means "this machine can't do CUDA" -
    /// callers surface that honestly and move on (CPU backend, other devices).
    pub fn probe(ordinal: usize) -> Result<Self, CudaError> {
        if !driver_present() {
            return Err(CudaError::Unavailable(NO_DRIVER.into()));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| CudaError::Unavailable(e.to_string()))?;

        let name = ctx.name().map_err(|e| CudaError::Driver(e.to_string()))?;
        use cudarc::driver::sys::CUdevice_attribute;
        let cc_major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let cc_minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let (_free, total) =
            cudarc::driver::result::mem_get_info().map_err(|e| CudaError::Driver(e.to_string()))?;

        Ok(Self {
            ctx,
            info: BackendInfo {
                name: "cuda",
                device: format!("{name} (SM {cc_major}.{cc_minor})"),
                memory_total: total as u64,
            },
        })
    }

    /// Free device memory right now - the estimator wants live numbers, not
    /// the total minus guesses.
    pub fn memory_free(&self) -> Result<u64, CudaError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        let (free, _total) =
            cudarc::driver::result::mem_get_info().map_err(|e| CudaError::Driver(e.to_string()))?;
        Ok(free as u64)
    }
}

impl Backend for CudaBackend {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    fn healthy(&self) -> bool {
        self.ctx.default_stream().synchronize().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs only where a CUDA device exists (dev box, GPU CI runner).
    #[test]
    fn probes_local_device_when_present() {
        match CudaBackend::probe(0) {
            Ok(be) => {
                let info = be.info();
                assert_eq!(info.name, "cuda");
                assert!(info.memory_total > 0);
                assert!(be.healthy());
                tracing::info!("probed: {} ({} bytes)", info.device, info.memory_total);
            }
            Err(e) => tracing::warn!("no CUDA here - skipping ({e})"),
        }
    }

    /// Without the driver library every entry point answers `Unavailable`
    /// naming the driver, and nothing panics. On a box that has the driver
    /// this only checks the probe agrees with `driver_present`.
    #[test]
    fn missing_driver_is_an_error_not_a_panic() {
        if driver_present() {
            assert!(CudaBackend::probe(0).is_ok() || resolve_device("0").is_ok());
            return;
        }
        match CudaBackend::probe(0) {
            Err(CudaError::Unavailable(msg)) => assert!(msg.contains("driver"), "{msg}"),
            other => panic!("expected Unavailable, got {:?}", other.map(|_| ())),
        }
        match resolve_device("GPU-d56cd6c9") {
            Err(CudaError::Unavailable(msg)) => assert!(msg.contains("driver"), "{msg}"),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }
}
