//! CUTLASS leaf-kernel provider.
//!
//! CUTLASS owns only kernel implementation and architecture-specific schedule
//! selection. Ferrule continues to own CUDA contexts, streams, allocations,
//! tensor lifetimes, execution plans, and fallback policy. The native boundary
//! is versioned POD data; no C++ object crosses into Rust.

#[cfg(feature = "cutlass")]
use cuda_core::{DeviceBuffer, stream::CudaStream};
#[cfg(feature = "cutlass")]
use ferrule_common::{Error, Result};

pub const CUTLASS_ABI_VERSION: u32 = 3;

/// Stable kernel IDs published by the native provider manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CutlassKernelId {
    F32Simt = 1,
    Bf16MmaSync = 2,
}

impl CutlassKernelId {
    pub const fn mask(self) -> u64 {
        1u64 << (self as u32 - 1)
    }
}

/// Native provider metadata used to reject ABI or target mismatches before a
/// kernel is selected into an executable plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassProviderManifest {
    pub abi_version: u32,
    /// CUTLASS's packed `major * 100 + minor * 10 + patch` version.
    pub cutlass_version: u32,
    pub target_sm: u32,
    pub kernel_count: u32,
    pub kernel_mask: u64,
}

impl CutlassProviderManifest {
    pub const fn supports(self, kernel: CutlassKernelId) -> bool {
        self.kernel_mask & kernel.mask() != 0
    }
}

/// Backward-compatible name for callers of the original metadata-only bridge.
pub type CutlassBuildInfo = CutlassProviderManifest;

/// POD launch descriptor for the first real CUTLASS provider kernel.
///
/// Computes `C = alpha * A * transpose(B) + beta * C`, where A is row-major
/// `[m, k]`, B is Ferrule's row-major weight matrix `[n, k]`, and C is
/// row-major `[m, n]` on a Ferrule-owned CUDA stream.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CutlassGemmF32Args {
    pub abi_version: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub stream: u64,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub reserved0: u32,
    pub alpha: f32,
    pub beta: f32,
    pub reserved1: u32,
    pub reserved2: u32,
}

/// POD descriptor for BF16 Tensor Core projection with F32 input/output and
/// caller-owned activation-pack workspace.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CutlassGemmBf16F32Args {
    pub abi_version: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_f32: u64,
    pub b_bf16: u64,
    pub c_f32: u64,
    pub workspace_bf16: u64,
    pub workspace_bytes: u64,
    pub stream: u64,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub reserved0: u32,
    pub alpha: f32,
    pub beta: f32,
    pub reserved1: u32,
    pub reserved2: u32,
}

/// Persistent BF16 activation-pack workspace owned by the Ferrule execution
/// image. CUTLASS receives only its pointer and capacity.
#[cfg(feature = "cutlass")]
pub struct CutlassBf16Workspace {
    buffer: DeviceBuffer<u8>,
    capacity_bytes: usize,
}

#[cfg(feature = "cutlass")]
impl CutlassBf16Workspace {
    pub fn new(stream: &CudaStream, max_rows: usize, k: usize) -> Result<Self> {
        let capacity_bytes = max_rows
            .checked_mul(k)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| Error::Internal("CUTLASS BF16 workspace size overflow".into()))?;
        if capacity_bytes == 0 {
            return Err(Error::Internal(
                "CUTLASS BF16 workspace cannot be empty".into(),
            ));
        }
        Ok(Self {
            buffer: DeviceBuffer::zeroed(stream, capacity_bytes).map_err(|error| {
                Error::Internal(format!("allocate CUTLASS BF16 workspace: {error:?}"))
            })?,
            capacity_bytes,
        })
    }

    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }
}

impl CutlassGemmF32Args {
    #[cfg(feature = "cutlass")]
    fn from_buffers(
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Self> {
        let a_len = m
            .checked_mul(k)
            .ok_or_else(|| Error::Internal("CUTLASS GEMM A size overflow".into()))?;
        let b_len = n
            .checked_mul(k)
            .ok_or_else(|| Error::Internal("CUTLASS GEMM B size overflow".into()))?;
        let c_len = m
            .checked_mul(n)
            .ok_or_else(|| Error::Internal("CUTLASS GEMM C size overflow".into()))?;
        if a.len() < a_len || b.len() < b_len || c.len() < c_len {
            return Err(Error::Internal(format!(
                "CUTLASS GEMM buffer mismatch: A={}/{a_len} B={}/{b_len} C={}/{c_len}",
                a.len(),
                b.len(),
                c.len()
            )));
        }
        if m == 0 || n == 0 || k == 0 || !alpha.is_finite() || !beta.is_finite() {
            return Err(Error::Internal(format!(
                "CUTLASS GEMM invalid problem: m={m} n={n} k={k} alpha={alpha} beta={beta}"
            )));
        }
        Ok(Self {
            abi_version: CUTLASS_ABI_VERSION,
            m: u32::try_from(m)
                .map_err(|_| Error::Internal("CUTLASS GEMM m exceeds u32".into()))?,
            n: u32::try_from(n)
                .map_err(|_| Error::Internal("CUTLASS GEMM n exceeds u32".into()))?,
            k: u32::try_from(k)
                .map_err(|_| Error::Internal("CUTLASS GEMM k exceeds u32".into()))?,
            a: a.cu_deviceptr(),
            b: b.cu_deviceptr(),
            c: c.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
            lda: u32::try_from(k)
                .map_err(|_| Error::Internal("CUTLASS GEMM lda exceeds u32".into()))?,
            ldb: u32::try_from(k)
                .map_err(|_| Error::Internal("CUTLASS GEMM ldb exceeds u32".into()))?,
            ldc: u32::try_from(n)
                .map_err(|_| Error::Internal("CUTLASS GEMM ldc exceeds u32".into()))?,
            reserved0: 0,
            alpha,
            beta,
            reserved1: 0,
            reserved2: 0,
        })
    }
}

impl CutlassGemmBf16F32Args {
    #[cfg(feature = "cutlass")]
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<u8>,
        c: &mut DeviceBuffer<f32>,
        workspace: &mut CutlassBf16Workspace,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Self> {
        let a_len = m
            .checked_mul(k)
            .ok_or_else(|| Error::Internal("CUTLASS BF16 GEMM A size overflow".into()))?;
        let b_len = n
            .checked_mul(k)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| Error::Internal("CUTLASS BF16 GEMM B size overflow".into()))?;
        let c_len = m
            .checked_mul(n)
            .ok_or_else(|| Error::Internal("CUTLASS BF16 GEMM C size overflow".into()))?;
        let workspace_bytes = a_len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::Internal("CUTLASS BF16 GEMM workspace overflow".into()))?;
        if a.len() < a_len
            || b.len() < b_len
            || c.len() < c_len
            || workspace.capacity_bytes < workspace_bytes
        {
            return Err(Error::Internal(format!(
                "CUTLASS BF16 GEMM buffer mismatch: A={}/{a_len} B={}/{b_len} C={}/{c_len} workspace={}/{workspace_bytes}",
                a.len(),
                b.len(),
                c.len(),
                workspace.capacity_bytes
            )));
        }
        if m == 0 || n == 0 || k == 0 || !alpha.is_finite() || !beta.is_finite() {
            return Err(Error::Internal(format!(
                "CUTLASS BF16 GEMM invalid problem: m={m} n={n} k={k} alpha={alpha} beta={beta}"
            )));
        }
        Ok(Self {
            abi_version: CUTLASS_ABI_VERSION,
            m: checked_u32(m, "m")?,
            n: checked_u32(n, "n")?,
            k: checked_u32(k, "k")?,
            a_f32: a.cu_deviceptr(),
            b_bf16: b.cu_deviceptr(),
            c_f32: c.cu_deviceptr(),
            workspace_bf16: workspace.buffer.cu_deviceptr(),
            workspace_bytes: workspace.capacity_bytes as u64,
            stream: stream.cu_stream() as usize as u64,
            lda: checked_u32(k, "lda")?,
            ldb: checked_u32(k, "ldb")?,
            ldc: checked_u32(n, "ldc")?,
            reserved0: 0,
            alpha,
            beta,
            reserved1: 0,
            reserved2: 0,
        })
    }
}

#[cfg(feature = "cutlass")]
fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Internal(format!("CUTLASS GEMM {name} exceeds u32")))
}

/// Whether this build contains the native CUTLASS provider.
pub const fn is_compiled() -> bool {
    cfg!(feature = "cutlass")
}

/// Return native provider metadata, or `None` when CUTLASS is disabled.
#[cfg(feature = "cutlass")]
pub fn provider_manifest() -> Option<CutlassProviderManifest> {
    let manifest = unsafe { ffi::ferrule_cutlass_provider_manifest() };
    (manifest.abi_version == CUTLASS_ABI_VERSION).then_some(manifest)
}

/// Return native provider metadata, or `None` when CUTLASS is disabled.
#[cfg(not(feature = "cutlass"))]
pub const fn provider_manifest() -> Option<CutlassProviderManifest> {
    None
}

/// Backward-compatible alias for the original bridge API.
pub fn build_info() -> Option<CutlassBuildInfo> {
    provider_manifest()
}

/// Convert the native catalog into Ferrule's provider registry vocabulary.
pub fn execution_provider_manifest() -> Option<ferrule_common::kernel_plan::ProviderManifest> {
    let manifest = provider_manifest()?;
    Some(
        ferrule_common::kernel_plan::ProviderManifest::cutlass_cubin(
            u16::try_from(manifest.abi_version).ok()?,
            manifest.kernel_count as usize,
        ),
    )
}

/// Ask the native provider whether its current kernel can implement a problem.
#[cfg(feature = "cutlass")]
pub fn gemm_f32_can_implement(
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    let args = CutlassGemmF32Args::from_buffers(stream, a, b, c, m, n, k, 1.0, 0.0)?;
    match unsafe { ffi::ferrule_cutlass_gemm_f32_can_implement(&args) } {
        status::SUCCESS => Ok(true),
        status::UNSUPPORTED => Ok(false),
        code => Err(native_error("can_implement f32 GEMM", code)),
    }
}

/// Launch CUTLASS F32 GEMM on a Ferrule-owned stream and allocation set.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32(
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    let args = CutlassGemmF32Args::from_buffers(stream, a, b, c, m, n, k, alpha, beta)?;
    let code = unsafe { ffi::ferrule_cutlass_gemm_f32_launch(&args) };
    if code == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch f32 GEMM", code))
    }
}

/// Ask whether the compiled BF16 Tensor Core specialization accepts the
/// problem and caller-owned workspace.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_f32_can_implement(
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<u8>,
    c: &mut DeviceBuffer<f32>,
    workspace: &mut CutlassBf16Workspace,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    let args = CutlassGemmBf16F32Args::from_buffers(stream, a, b, c, workspace, m, n, k, 1.0, 0.0)?;
    match unsafe { ffi::ferrule_cutlass_gemm_bf16_f32_can_implement(&args) } {
        status::SUCCESS => Ok(true),
        status::UNSUPPORTED => Ok(false),
        code => Err(native_error("can_implement BF16 GEMM", code)),
    }
}

/// Pack F32 activations into caller-owned BF16 workspace and launch the BF16
/// Tensor Core projection on the same Ferrule stream.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_f32(
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<u8>,
    c: &mut DeviceBuffer<f32>,
    workspace: &mut CutlassBf16Workspace,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    let args =
        CutlassGemmBf16F32Args::from_buffers(stream, a, b, c, workspace, m, n, k, alpha, beta)?;
    let code = unsafe { ffi::ferrule_cutlass_gemm_bf16_f32_launch(&args) };
    if code == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch BF16 GEMM", code))
    }
}

#[cfg(feature = "cutlass")]
fn native_error(operation: &str, code: i32) -> Error {
    let reason = match code {
        status::INVALID_ABI => "ABI mismatch",
        status::INVALID_ARGUMENT => "invalid argument",
        status::UNSUPPORTED => "unsupported problem",
        status::LAUNCH_FAILED => "kernel launch failed",
        _ => "unknown native status",
    };
    Error::Internal(format!("CUTLASS {operation} failed: {reason} ({code})"))
}

#[cfg(feature = "cutlass")]
mod status {
    pub const SUCCESS: i32 = 0;
    pub const INVALID_ABI: i32 = 1;
    pub const INVALID_ARGUMENT: i32 = 2;
    pub const UNSUPPORTED: i32 = 3;
    pub const LAUNCH_FAILED: i32 = 4;
}

#[cfg(feature = "cutlass")]
mod ffi {
    use super::{CutlassGemmBf16F32Args, CutlassGemmF32Args, CutlassProviderManifest};

    unsafe extern "C" {
        pub fn ferrule_cutlass_provider_manifest() -> CutlassProviderManifest;
        pub fn ferrule_cutlass_gemm_f32_can_implement(args: *const CutlassGemmF32Args) -> i32;
        pub fn ferrule_cutlass_gemm_f32_launch(args: *const CutlassGemmF32Args) -> i32;
        pub fn ferrule_cutlass_gemm_bf16_f32_can_implement(
            args: *const CutlassGemmBf16F32Args,
        ) -> i32;
        pub fn ferrule_cutlass_gemm_bf16_f32_launch(args: *const CutlassGemmBf16F32Args) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_layout_matches_native_contract() {
        assert_eq!(std::mem::size_of::<CutlassProviderManifest>(), 24);
        assert_eq!(std::mem::size_of::<CutlassGemmF32Args>(), 80);
        assert_eq!(std::mem::size_of::<CutlassGemmBf16F32Args>(), 96);
    }

    #[test]
    fn build_info_matches_feature_state() {
        assert_eq!(build_info().is_some(), is_compiled());
    }

    #[cfg(feature = "cutlass")]
    #[test]
    fn native_provider_matches_pinned_abi() {
        let manifest = provider_manifest().expect("CUTLASS provider manifest");
        assert_eq!(manifest.abi_version, CUTLASS_ABI_VERSION);
        assert_eq!(manifest.cutlass_version, 461);
        assert_eq!(manifest.kernel_count, 2);
        assert!(manifest.supports(CutlassKernelId::F32Simt));
        assert!(manifest.supports(CutlassKernelId::Bf16MmaSync));
        assert!(manifest.target_sm >= 80);
    }
}
