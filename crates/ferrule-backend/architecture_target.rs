//! Shared CUDA target parsing used by `build.rs` and the CUDA runtime.

/// CUDA architecture generation. Instruction availability is represented by
/// [`CudaKernelCapabilities`], not inferred from this coarse family enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArchitectureFamily {
    Ampere,
    Ada,
    Hopper,
    Blackwell,
    Unknown,
}

/// Parsed device target such as `sm_86`, `sm_103`, or `compute_121`.
///
/// Compiler-only feature scopes are selected privately by `build.rs`; they are
/// not device architectures and therefore do not belong in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaTarget {
    pub major: u32,
    pub minor: u32,
    pub family: CudaArchitectureFamily,
}

impl CudaTarget {
    pub fn parse(value: &str) -> Option<Self> {
        let target = value
            .strip_prefix("sm_")
            .or_else(|| value.strip_prefix("compute_"))?;
        if target.len() < 2 || !target.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let split = target.len() - 1;
        let major = target[..split].parse().ok()?;
        let minor = target[split..].parse().ok()?;
        let family = match (major, minor) {
            (8, 9) => CudaArchitectureFamily::Ada,
            (8, _) => CudaArchitectureFamily::Ampere,
            (9, _) => CudaArchitectureFamily::Hopper,
            (10..=12, _) => CudaArchitectureFamily::Blackwell,
            _ => CudaArchitectureFamily::Unknown,
        };
        Some(Self {
            major,
            minor,
            family,
        })
    }

    pub const fn compute_capability(self) -> u32 {
        self.major * 10 + self.minor
    }

    pub const fn capabilities(self) -> CudaKernelCapabilities {
        let compute_capability = self.compute_capability();
        CudaKernelCapabilities {
            portable_simt: compute_capability >= 80,
            bf16_mma_sync: compute_capability >= 80,
            fp8_mma_sync: compute_capability >= 89,
            sm90_wgmma: compute_capability == 90,
            sm1xx_umma: matches!(compute_capability, 100 | 101 | 110),
            sm103_block_scaled_fp4: compute_capability == 103,
            sm12x_mxfp4_mma_sync: matches!(compute_capability, 120 | 121),
        }
    }
}

/// Native instruction families available to a provider compiled for one exact
/// target. Product names and scheduling policies do not belong in this type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaKernelCapabilities {
    /// Portable control, metadata, scalar/SIMT, and software-dequant kernels.
    pub portable_simt: bool,
    /// Ampere-or-newer BF16 `mma.sync` kernels.
    pub bf16_mma_sync: bool,
    /// Ada-or-newer FP8 `mma.sync` kernels.
    pub fp8_mma_sync: bool,
    /// Hopper architecture-specific WGMMA instructions.
    pub sm90_wgmma: bool,
    /// Architecture-feature UMMA/tensor-memory instructions for SM100/101/110.
    pub sm1xx_umma: bool,
    /// SM103 architecture-specific block-scaled FP4 tensor-core collectives.
    pub sm103_block_scaled_fp4: bool,
    /// SM120/SM121 warp-level block-scaled FP4 instructions.
    pub sm12x_mxfp4_mma_sync: bool,
}
