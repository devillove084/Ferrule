//! CUDA architecture and kernel-ISA capabilities frozen into the module.
//!
//! `cargo oxide --arch ...` exports the selected target through
//! `CUDA_OXIDE_TARGET`. The build script parses it with the same code used here
//! and emits cfgs for concrete kernel instruction families. Provider selection
//! therefore distinguishes portable SIMT/BF16, Hopper WGMMA, datacenter
//! Blackwell tcgen05, and GB10-style Blackwell `mma.sync` kernels.

#[path = "../architecture_target.rs"]
mod target;

pub use target::{CudaArchitectureFamily, CudaKernelCapabilities, CudaTarget, CudaTargetSuffix};

/// CUDA target string selected when this crate was compiled.
pub const COMPILED_TARGET: &str = env!("FERRULE_CUDA_COMPILED_TARGET");

/// Current GB10 FP8 `mma.sync` provider was included in the module.
pub const COMPILED_BLACKWELL_MMA_SYNC_FP8: bool = cfg!(ferrule_cuda_blackwell_mma_sync_fp8);

/// Current GB10 MXFP4 `mma.sync` provider was included in the module.
pub const COMPILED_BLACKWELL_MMA_SYNC_MXFP4: bool = cfg!(ferrule_cuda_blackwell_mma_sync_mxfp4);

/// cuda-oxide's BF16 MMA intrinsic is compiled only for the modern NVVM path.
/// Ampere/Hopper builds use the CUTLASS provider instead of emitting an
/// unresolved legacy-NVVM device symbol.
pub const COMPILED_CUDA_OXIDE_BF16_MMA: bool = cfg!(ferrule_cuda_cuda_oxide_bf16_mma);

/// Parse the compiled target and return every architecture capability, while
/// masking native providers that are represented but not yet compiled.
pub fn compiled_capabilities() -> CudaKernelCapabilities {
    let mut capabilities = CudaTarget::parse(COMPILED_TARGET)
        .map(CudaTarget::capabilities)
        .unwrap_or_default();
    capabilities.blackwell_mma_sync_fp8 &= COMPILED_BLACKWELL_MMA_SYNC_FP8;
    capabilities.blackwell_mma_sync_mxfp4 &= COMPILED_BLACKWELL_MMA_SYNC_MXFP4;
    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_architecture_families() {
        assert_eq!(
            CudaTarget::parse("sm_86").unwrap().family,
            CudaArchitectureFamily::Ampere
        );
        assert_eq!(
            CudaTarget::parse("sm_90a").unwrap().family,
            CudaArchitectureFamily::Hopper
        );
        assert_eq!(
            CudaTarget::parse("sm_100a").unwrap().family,
            CudaArchitectureFamily::BlackwellDatacenter
        );
        assert_eq!(
            CudaTarget::parse("sm_121a").unwrap().family,
            CudaArchitectureFamily::BlackwellConsumer
        );
    }

    #[test]
    fn keeps_native_instruction_families_distinct() {
        let h200 = CudaTarget::parse("sm_90a").unwrap().capabilities();
        assert!(h200.hopper_wgmma);
        assert!(!h200.blackwell_tcgen05);
        assert!(!h200.blackwell_mma_sync_fp8);

        let b200 = CudaTarget::parse("sm_100a").unwrap().capabilities();
        assert!(!b200.hopper_wgmma);
        assert!(b200.blackwell_tcgen05);
        assert!(!b200.blackwell_mma_sync_fp8);

        let gb10 = CudaTarget::parse("sm_121a").unwrap().capabilities();
        assert!(!gb10.hopper_wgmma);
        assert!(!gb10.blackwell_tcgen05);
        assert!(gb10.blackwell_mma_sync_fp8);
        assert!(gb10.blackwell_mma_sync_mxfp4);
    }

    #[test]
    fn ampere_keeps_portable_and_bf16_paths() {
        let capabilities = CudaTarget::parse("sm_86").unwrap().capabilities();
        assert!(capabilities.portable_simt);
        assert!(capabilities.bf16_mma_sync);
        assert!(!capabilities.hopper_wgmma);
        assert!(!capabilities.blackwell_tcgen05);
        assert!(!capabilities.blackwell_mma_sync_fp8);
    }

    #[test]
    fn malformed_target_has_no_capabilities() {
        assert!(CudaTarget::parse("portable").is_none());
        assert!(CudaTarget::parse("sm_xx").is_none());
    }
}
