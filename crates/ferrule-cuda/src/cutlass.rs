//! Optional CUTLASS SM121a provider bridge.
//!
//! Cargo compiles the native bridge only when the `cutlass-sm121a` feature is
//! enabled. The bridge consumes a pinned CUTLASS header checkout but does not
//! create CUDA contexts, streams, or allocations.

/// Native bridge metadata used to reject ABI or toolchain mismatches early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutlassBuildInfo {
    pub abi_version: u32,
    /// CUTLASS's packed `major * 100 + minor * 10 + patch` version.
    pub cutlass_version: u32,
    pub target_sm: u32,
}

/// Whether this build contains the optional SM121a bridge.
pub const fn is_compiled() -> bool {
    cfg!(feature = "cutlass-sm121a")
}

/// Return native bridge metadata, or `None` when the feature is disabled.
#[cfg(feature = "cutlass-sm121a")]
pub fn build_info() -> Option<CutlassBuildInfo> {
    Some(CutlassBuildInfo {
        // SAFETY: these functions have no arguments, return scalar values, and
        // are linked from the bridge compiled by this crate's build script.
        abi_version: unsafe { ffi::ferrule_cutlass_abi_version() },
        cutlass_version: unsafe { ffi::ferrule_cutlass_version() },
        target_sm: unsafe { ffi::ferrule_cutlass_target_sm() },
    })
}

/// Return native bridge metadata, or `None` when the feature is disabled.
#[cfg(not(feature = "cutlass-sm121a"))]
pub const fn build_info() -> Option<CutlassBuildInfo> {
    None
}

#[cfg(feature = "cutlass-sm121a")]
mod ffi {
    unsafe extern "C" {
        pub fn ferrule_cutlass_abi_version() -> u32;
        pub fn ferrule_cutlass_version() -> u32;
        pub fn ferrule_cutlass_target_sm() -> u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_matches_feature_state() {
        assert_eq!(build_info().is_some(), is_compiled());
    }

    #[cfg(feature = "cutlass-sm121a")]
    #[test]
    fn native_bridge_is_the_pinned_sm121a_build() {
        assert_eq!(
            build_info(),
            Some(CutlassBuildInfo {
                abi_version: 1,
                cutlass_version: 461,
                target_sm: 121,
            })
        );
    }
}
