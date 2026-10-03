//! One way in to NVML, because the obvious way is wrong on Linux.
//!
//! `Nvml::init()` loads `libnvidia-ml.so` on Linux - the UNVERSIONED name. The
//! NVIDIA driver installs `libnvidia-ml.so.1`; the bare `.so` is a development
//! symlink that comes with the headers package and is absent from every
//! runtime-only install, which includes every container the NVIDIA Container
//! Toolkit builds.
//!
//! Seen on a Linux release under `--gpus all` with a
//! working A6000: the manager announced "no usable NVIDIA graphics card found
//! - models cannot run on this computer". `nvidia-smi` worked in the same
//!   container, `libnvidia-ml.so.1` was in the loader cache, and adding the one
//!   symlink flipped it to "graphics card supported". So the product told a user
//!   with a supported card that their card did not exist - a confidently wrong
//!   answer, which is worse than the silent failure the principles already ban.
//!
//! Windows is unaffected: there the name is `nvml.dll`, versionless, and
//! present whenever the driver is. Hence a fallback rather than a replacement
//! - the default is right on one platform and right on developer Linux boxes
//!   too, and this only catches the case where it is not.
use nvml_wrapper::{Device, Nvml, error::NvmlError};

/// Initialise NVML, looking under the versioned soname if the default name is
/// not installed.
pub fn init() -> Result<Nvml, NvmlError> {
    match Nvml::init() {
        Ok(n) => Ok(n),
        // Keep the first error to report: it names the library a reader will
        // recognise, and on a box with no driver at all both attempts fail
        // for the same reason anyway.
        Err(first) => {
            for name in FALLBACKS {
                if let Ok(n) = Nvml::builder().lib_path(name.as_ref()).init() {
                    return Ok(n);
                }
            }
            Err(first)
        }
    }
}

#[cfg(unix)]
const FALLBACKS: &[&str] = &["libnvidia-ml.so.1"];

#[cfg(not(unix))]
const FALLBACKS: &[&str] = &[];

/// Whether a device computes in the HOST's memory: an ATS-addressed part for
/// which NVML reports no framebuffer - the DGX Spark's GB10 ("FB Memory Usage:
/// N/A", "Addressing Mode: ATS" in `nvidia-smi -q`). Its memory is the
/// machine's RAM, so the sampler reads it from the OS instead
/// (`paddock_models::meminfo`, the reading the runner's load gate takes).
///
/// Asked only once `memory_info()` has said NotSupported, and answered no on
/// a driver too old to know the call, so a discrete card - whose framebuffer
/// always reports, HMM-addressed or not - can never be read as host memory.
pub fn host_memory_device(nvml: &Nvml, device: &Device<'_>) -> bool {
    use nvml_wrapper_sys::bindings::{
        nvmlDeviceAddressingMode_t, nvmlDeviceAddressingModeType_t_NVML_DEVICE_ADDRESSING_MODE_ATS,
        nvmlReturn_enum_NVML_SUCCESS,
    };
    let lib = nvml.lib();
    if lib.nvmlDeviceGetAddressingMode.is_err() {
        return false; // a driver older than the call
    }
    // NVML_STRUCT_VERSION(DeviceAddressingMode, 1): the struct's size, the
    // version in the top byte
    let mut mode = nvmlDeviceAddressingMode_t {
        version: std::mem::size_of::<nvmlDeviceAddressingMode_t>() as u32 | (1 << 24),
        value: 0,
    };
    // SAFETY: the symbol was resolved (checked above); the handle is live for
    // as long as `device` borrows `nvml`; `mode` is a v1 struct NVML fills.
    let r = unsafe { lib.nvmlDeviceGetAddressingMode(device.handle(), &mut mode) };
    r == nvmlReturn_enum_NVML_SUCCESS
        && mode.value == nvmlDeviceAddressingModeType_t_NVML_DEVICE_ADDRESSING_MODE_ATS
}
