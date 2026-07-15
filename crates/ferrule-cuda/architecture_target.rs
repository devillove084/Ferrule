//! Shared CUDA target parsing used by both `build.rs` and the runtime crate.

/// CUDA architecture family. This deliberately does not encode a kernel
/// implementation: H200, B200, and GB10 use different native instruction
/// families even though all are valid Ferrule targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArchitectureFamily {
    Ampere,
    Ada,
    Hopper,
    BlackwellDatacenter,
    BlackwellConsumer,
    Unknown,
}

/// PTX target suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaTargetSuffix {
    Plain,
    Accelerated,
    Family,
}

/// Parsed `sm_XX`, `sm_XXX[a|f]`, or `compute_XX` target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaTarget {
    pub major: u32,
    pub minor: u32,
    pub suffix: CudaTargetSuffix,
    pub family: CudaArchitectureFamily,
}

impl CudaTarget {
    pub fn parse(value: &str) -> Option<Self> {
        let target = value
            .strip_prefix("sm_")
            .or_else(|| value.strip_prefix("compute_"))?;
        let (digits, suffix) = match target.as_bytes().last().copied() {
            Some(b'a') => (&target[..target.len() - 1], CudaTargetSuffix::Accelerated),
            Some(b'f') => (&target[..target.len() - 1], CudaTargetSuffix::Family),
            _ => (target, CudaTargetSuffix::Plain),
        };
        if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let split = digits.len() - 1;
        let major = digits[..split].parse().ok()?;
        let minor = digits[split..].parse().ok()?;
        let family = match (major, minor) {
            (8, 9) => CudaArchitectureFamily::Ada,
            (8, _) => CudaArchitectureFamily::Ampere,
            (9, _) => CudaArchitectureFamily::Hopper,
            (10 | 11, _) => CudaArchitectureFamily::BlackwellDatacenter,
            (12, _) => CudaArchitectureFamily::BlackwellConsumer,
            _ => CudaArchitectureFamily::Unknown,
        };
        Some(Self {
            major,
            minor,
            suffix,
            family,
        })
    }

    pub const fn compute_capability(self) -> u32 {
        self.major * 10 + self.minor
    }

    pub const fn has_accelerated_target(self) -> bool {
        matches!(self.suffix, CudaTargetSuffix::Accelerated)
    }

    pub const fn capabilities(self) -> CudaKernelCapabilities {
        let compute_capability = self.compute_capability();
        let accelerated = self.has_accelerated_target();
        CudaKernelCapabilities {
            portable_simt: compute_capability >= 80,
            bf16_mma_sync: compute_capability >= 80,
            hopper_wgmma: matches!(self.family, CudaArchitectureFamily::Hopper) && accelerated,
            blackwell_tcgen05: matches!(self.family, CudaArchitectureFamily::BlackwellDatacenter)
                && accelerated,
            blackwell_mma_sync_fp8: matches!(
                self.family,
                CudaArchitectureFamily::BlackwellConsumer
            ) && accelerated,
            blackwell_mma_sync_mxfp4: matches!(
                self.family,
                CudaArchitectureFamily::BlackwellConsumer
            ) && accelerated,
        }
    }
}

/// Native instruction families available to providers compiled for a target.
/// Fields for H200 and B200 are represented before their providers are wired so
/// provider selection never conflates them with the GB10 `mma.sync` kernels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaKernelCapabilities {
    /// Portable control, metadata, scalar/SIMT, and software-dequant kernels.
    pub portable_simt: bool,
    /// Ampere-or-newer BF16 `mma.sync` kernels.
    pub bf16_mma_sync: bool,
    /// Hopper/H200 WGMMA provider capability.
    pub hopper_wgmma: bool,
    /// Datacenter Blackwell/B200 tcgen05 provider capability.
    pub blackwell_tcgen05: bool,
    /// Consumer/SoC Blackwell FP8 `mma.sync` used by the GB10 provider.
    pub blackwell_mma_sync_fp8: bool,
    /// Consumer/SoC Blackwell MXFP4 `mma.sync` used by the GB10 provider.
    pub blackwell_mma_sync_mxfp4: bool,
}
