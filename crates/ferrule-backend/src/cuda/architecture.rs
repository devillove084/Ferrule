//! CUDA architecture and native ISA capabilities frozen into this build.

#[path = "../../architecture_target.rs"]
mod target;

pub use target::{CudaArchitectureFamily, CudaKernelCapabilities, CudaTarget};

/// CUDA target string selected when this crate was compiled.
pub const COMPILED_TARGET: &str = env!("FERRULE_BACKEND_CUDA_COMPILED_TARGET");

/// Parse the compiled target and return its exact ISA capabilities.
pub fn compiled_capabilities() -> CudaKernelCapabilities {
    CudaTarget::parse(COMPILED_TARGET)
        .map(CudaTarget::capabilities)
        .unwrap_or_default()
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
            CudaTarget::parse("sm_90").unwrap().family,
            CudaArchitectureFamily::Hopper
        );
        assert_eq!(
            CudaTarget::parse("sm_103").unwrap().family,
            CudaArchitectureFamily::Blackwell
        );
        assert_eq!(
            CudaTarget::parse("sm_121").unwrap().family,
            CudaArchitectureFamily::Blackwell
        );
    }

    #[test]
    fn sm103_publishes_its_block_scaled_fp4_capability() {
        let sm103 = CudaTarget::parse("sm_103").unwrap().capabilities();
        assert!(sm103.portable_simt);
        assert!(sm103.bf16_mma_sync);
        assert!(sm103.fp8_mma_sync);
        assert!(!sm103.sm1xx_umma);
        assert!(sm103.sm103_block_scaled_fp4);
        assert!(!sm103.sm12x_mxfp4_mma_sync);
    }

    #[test]
    fn instruction_families_do_not_follow_product_buckets() {
        let hopper = CudaTarget::parse("sm_90").unwrap().capabilities();
        assert!(hopper.sm90_wgmma);
        assert!(!hopper.sm1xx_umma);
        assert!(!hopper.sm103_block_scaled_fp4);

        let sm121 = CudaTarget::parse("sm_121").unwrap().capabilities();
        assert!(!sm121.sm90_wgmma);
        assert!(!sm121.sm1xx_umma);
        assert!(!sm121.sm103_block_scaled_fp4);
        assert!(sm121.sm12x_mxfp4_mma_sync);
    }

    #[test]
    fn malformed_target_has_no_capabilities() {
        assert!(CudaTarget::parse("portable").is_none());
        assert!(CudaTarget::parse("sm_xx").is_none());
        assert!(CudaTarget::parse("sm_103a").is_none());
        assert!(CudaTarget::parse("sm_121f").is_none());
    }
}
