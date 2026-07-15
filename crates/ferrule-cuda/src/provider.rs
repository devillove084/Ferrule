//! CUDA kernel provider boundary.
//!
//! This module re-exports the provider-agnostic kernel plan types from
//! `ferrule_common::kernel_plan` and will host CUDA-specific provider
//! implementations (cubin loading, `LoadedModule` management) as the
//! superkernel bundles are implemented.
//!
//! See `ferrule_common::kernel_plan` for the canonical type definitions.

pub use ferrule_common::kernel_plan::{
    KernelId, KernelPhase, KernelProviderId, LaunchDescriptor, LayerKernelPlan, LayerKernelPlanSet,
    ModelKernelPlan, ProviderManifest, ProviderRegistry, RowBucket, WeightBinding, WeightLayout,
    compile_default_plan,
};
