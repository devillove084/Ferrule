//! Semantic CUDA device diagnostics.

use ferrule_common::Result;

use crate::cuda::runtime::CudaContext;

/// Memory capacity reported for a CUDA device context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryDiagnostics {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// Typed diagnostics for one usable CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceProbe {
    pub ordinal: usize,
    pub name: String,
    pub memory: CudaMemoryDiagnostics,
}

/// Create the device's primary context and collect its semantic diagnostics.
pub fn probe_device(ordinal: usize) -> Result<CudaDeviceProbe> {
    let context = CudaContext::new(ordinal)?;
    let name = context.device_name()?;
    let (free_bytes, total_bytes) = context.memory_info()?;

    Ok(CudaDeviceProbe {
        ordinal,
        name,
        memory: CudaMemoryDiagnostics {
            free_bytes,
            total_bytes,
        },
    })
}
