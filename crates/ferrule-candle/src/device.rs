use candle_core::Device;
use ferrule_core::{FerruleError, FerruleResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
}

pub fn select_device(spec: &str) -> FerruleResult<Device> {
    match spec {
        "cpu" => Ok(Device::Cpu),
        "auto" => auto_device(),
        s if s.starts_with("cuda:") => {
            let idx = parse_index(s, "cuda:")?;
            cuda_device(idx)
        }
        s if s.starts_with("metal:") => {
            let idx = parse_index(s, "metal:")?;
            metal_device(idx)
        }
        other => Err(FerruleError::Config(format!(
            "unsupported device spec: {other}"
        ))),
    }
}

pub fn device_kind(device: &Device) -> DeviceKind {
    if device.is_cuda() {
        DeviceKind::Cuda
    } else if device.is_metal() {
        DeviceKind::Metal
    } else {
        DeviceKind::Cpu
    }
}

pub fn device_kind_str(device: &Device) -> &'static str {
    match device_kind(device) {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Cuda => "cuda",
        DeviceKind::Metal => "metal",
    }
}

pub fn compiled_backends_summary() -> String {
    format!(
        "compiled_backends(cuda={}, metal={})",
        candle_core::utils::cuda_is_available(),
        candle_core::utils::metal_is_available(),
    )
}

fn parse_index(spec: &str, prefix: &str) -> FerruleResult<usize> {
    let raw = spec.trim_start_matches(prefix);
    raw.parse::<usize>()
        .map_err(|e| FerruleError::Config(format!("invalid device index in '{spec}': {e}")))
}

fn auto_device() -> FerruleResult<Device> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }

    Ok(Device::Cpu)
}

fn cuda_device(idx: usize) -> FerruleResult<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(idx)
            .map_err(|e| FerruleError::Setup(format!("failed to init cuda:{idx}: {e}")));
    }

    #[allow(unreachable_code)]
    Err(FerruleError::Setup(
        "cuda backend is not enabled for ferrule-candle".to_string(),
    ))
}

fn metal_device(idx: usize) -> FerruleResult<Device> {
    #[cfg(feature = "metal")]
    {
        return Device::new_metal(idx)
            .map_err(|e| FerruleError::Setup(format!("failed to init metal:{idx}: {e}")));
    }

    #[allow(unreachable_code)]
    Err(FerruleError::Setup(
        "metal backend is not enabled for ferrule-candle".to_string(),
    ))
}
