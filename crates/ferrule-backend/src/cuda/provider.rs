//! CUDA kernel-provider discovery and executable-plan compilation.
//!
//! Provider selection happens during model preparation. The hot path consumes
//! resolved [`LaunchDescriptor`] values and never queries architecture strings,
//! environment variables, or dynamic trait objects.

use ferrule_common::Result;

use crate::BackendError;
use crate::cuda::cutlass::{CutlassKernelId, CutlassProviderManifest};

pub use crate::plan::{
    ExecutionMode, ExecutionModeSet, KernelId, KernelOperation, KernelPhase, KernelProviderId,
    LaunchDescriptor, LayerKernelPlan, LayerKernelRequirements, LinearBundleRequirement,
    ModelKernelPlan, OperationCapability, OperationRequirement, ProviderManifest, ProviderRegistry,
    WeightBinding, WeightLayout,
};

/// Providers and native catalogs available in this build.
#[derive(Debug, Clone)]
pub struct CudaProviderCatalog {
    registry: ProviderRegistry,
    cutlass: CutlassProviderManifest,
}

impl CudaProviderCatalog {
    /// Discover compiled providers once during prepare/compile.
    ///
    /// The native manifest's kernel mask is the sole source of operation
    /// capabilities. Rust-side target metadata does not add capabilities.
    pub fn discover() -> Result<Self> {
        let cutlass = native_cutlass_manifest();
        let mut registry = ProviderRegistry::new();
        registry.register(cutlass_execution_manifest(cutlass));
        Ok(Self { registry, cutlass })
    }

    pub const fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    pub const fn cutlass_manifest(&self) -> CutlassProviderManifest {
        self.cutlass
    }

    /// Compile provider-neutral requirements into one semantic plan per layer.
    /// Native providers own row-count schedule dispatch. Missing capabilities
    /// are fatal; production has no fallback path.
    pub fn compile_model_plan(
        &self,
        requirements: &[LayerKernelRequirements],
    ) -> Result<ModelKernelPlan> {
        let mut model_plan = ModelKernelPlan::new(requirements.len());

        for (layer, requirements) in requirements.iter().enumerate() {
            let layer_plan =
                model_plan
                    .layer_mut(layer)
                    .ok_or_else(|| BackendError::Invariant {
                        message: format!("kernel plan lost layer slot {layer}"),
                    })?;
            for requirement in &requirements.linear_bundles {
                self.require_operation(OperationRequirement::new(
                    requirement.operation,
                    requirement.mode,
                ))?;
                require_cutlass_bundle(self.cutlass, requirement)?;
                set_provider_operation(layer_plan, requirement.operation, requirement.mode);
            }
            for &requirement in &requirements.operations {
                self.require_operation(requirement)?;
                require_semantic_operation(self.cutlass, requirement.operation)?;
                set_provider_operation(layer_plan, requirement.operation, requirement.mode);
            }
        }
        Ok(model_plan)
    }

    fn require_operation(&self, requirement: OperationRequirement) -> Result<()> {
        if !self
            .registry
            .supports_requirement(KernelProviderId::CUDA_CUTLASS, requirement)
        {
            return Err(BackendError::UnsupportedOperation {
                provider: KernelProviderId::CUDA_CUTLASS,
                operation: requirement.operation,
                mode: requirement.mode,
                deterministic: requirement.deterministic,
            }
            .into());
        }
        Ok(())
    }
}

fn set_provider_operation(
    plan: &mut LayerKernelPlan,
    operation: KernelOperation,
    mode: ExecutionMode,
) {
    let kernel = KernelId::new(KernelProviderId::CUDA_CUTLASS, operation, mode);
    let descriptor = LaunchDescriptor::new(kernel, (0, 0, 0), (0, 0, 0))
        .provider_managed()
        .capture_safe();
    plan.set_operation(descriptor);
}

fn require_semantic_operation(
    manifest: CutlassProviderManifest,
    operation: KernelOperation,
) -> Result<()> {
    let kernel = cutlass_kernel(operation).ok_or_else(|| BackendError::Invariant {
        message: format!(
            "CUDA CUTLASS manifest published {operation:?} without a semantic binding"
        ),
    })?;
    require_kernel(manifest, operation, kernel)
}

fn require_cutlass_bundle(
    manifest: CutlassProviderManifest,
    requirement: &LinearBundleRequirement,
) -> Result<()> {
    let reject = |reason: String| BackendError::UnsupportedLinearBundle {
        provider: KernelProviderId::CUDA_CUTLASS,
        operation: requirement.operation,
        mode: requirement.mode,
        layout: requirement.weight_layout,
        reason,
    };

    if requirement.output_features.is_empty() || requirement.output_features.contains(&0) {
        return Err(reject("output feature list must be non-empty and non-zero".into()).into());
    }

    let kernel = match (requirement.operation, requirement.weight_layout) {
        (KernelOperation::MlaQueryAKv, WeightLayout::Fp8E4m3BlockScaled) => {
            if requirement.output_features.len() != 2
                || !requirement.input_features.is_multiple_of(128)
            {
                return Err(reject(format!(
                    "QueryA+KV requires two outputs and K128, got K={} N={:?}",
                    requirement.input_features, requirement.output_features
                ))
                .into());
            }
            CutlassKernelId::Fp8QueryAKv
        }
        (KernelOperation::MlaQueryB, WeightLayout::Fp8E4m3BlockScaled) => {
            if requirement.output_features.len() != 1
                || !requirement.input_features.is_multiple_of(128)
            {
                return Err(reject(format!(
                    "QueryB requires one output and K128, got K={} N={:?}",
                    requirement.input_features, requirement.output_features
                ))
                .into());
            }
            CutlassKernelId::Fp8Projection
        }
        (
            KernelOperation::MainCompressorProjection
            | KernelOperation::IndexerCompressorProjection,
            WeightLayout::Bf16RowMajor,
        ) => {
            if requirement.output_features.len() != 2
                || !requirement.input_features.is_multiple_of(8)
                || !requirement
                    .output_features
                    .iter()
                    .all(|features| features.is_multiple_of(4))
            {
                return Err(reject(format!(
                    "BF16 compressor requires two N4 outputs and K8, got K={} N={:?}",
                    requirement.input_features, requirement.output_features
                ))
                .into());
            }
            CutlassKernelId::Bf16Compressor
        }
        _ => {
            return Err(
                reject("operation and weight layout have no provider binding".into()).into(),
            );
        }
    };
    require_kernel(manifest, requirement.operation, kernel)
}

fn require_kernel(
    manifest: CutlassProviderManifest,
    operation: KernelOperation,
    kernel: CutlassKernelId,
) -> Result<()> {
    if !manifest.supports(kernel) {
        return Err(BackendError::MissingNativeKernel {
            provider: KernelProviderId::CUDA_CUTLASS,
            operation,
            kernel: kernel.name(),
        }
        .into());
    }
    Ok(())
}

const CUTLASS_OPERATIONS: [KernelOperation; 12] = [
    KernelOperation::MlaQueryAKv,
    KernelOperation::MlaQueryB,
    KernelOperation::MainCompressorProjection,
    KernelOperation::IndexerCompressorProjection,
    KernelOperation::AttentionHcPre,
    KernelOperation::FeedForwardHcPre,
    KernelOperation::MlaOutput,
    KernelOperation::SharedFfn,
    KernelOperation::GroupedFp4Moe,
    KernelOperation::MainProjectNorm,
    KernelOperation::HybridMlaAttention,
    KernelOperation::ProposalHead,
];

fn cutlass_execution_manifest(native: CutlassProviderManifest) -> ProviderManifest {
    let operations = CUTLASS_OPERATIONS
        .into_iter()
        .filter_map(|operation| {
            let kernel = cutlass_kernel(operation)?;
            native.supports(kernel).then_some(OperationCapability::new(
                operation,
                ExecutionModeSet::INFERENCE,
            ))
        })
        .collect::<Vec<_>>();

    ProviderManifest::new(KernelProviderId::CUDA_CUTLASS, "cuda-cutlass", operations)
}

const fn cutlass_kernel(operation: KernelOperation) -> Option<CutlassKernelId> {
    match operation {
        KernelOperation::MlaQueryAKv => Some(CutlassKernelId::Fp8QueryAKv),
        KernelOperation::MlaQueryB => Some(CutlassKernelId::Fp8Projection),
        KernelOperation::MainCompressorProjection
        | KernelOperation::IndexerCompressorProjection => Some(CutlassKernelId::Bf16Compressor),
        KernelOperation::AttentionHcPre | KernelOperation::FeedForwardHcPre => {
            Some(CutlassKernelId::HyperConnectionProducer)
        }
        KernelOperation::MlaOutput => Some(CutlassKernelId::MlaOutput),
        KernelOperation::SharedFfn => Some(CutlassKernelId::SharedFfn),
        KernelOperation::GroupedFp4Moe => Some(CutlassKernelId::GroupedFp4Moe),
        KernelOperation::MainProjectNorm => Some(CutlassKernelId::MainProjectNorm),
        KernelOperation::HybridMlaAttention => Some(CutlassKernelId::HybridMlaAttention),
        KernelOperation::ProposalHead => Some(CutlassKernelId::ProposalHead),
        _ => None,
    }
}

fn native_cutlass_manifest() -> CutlassProviderManifest {
    // SAFETY: the symbol is linked from the native provider and returns a POD
    // manifest by value without borrowing Rust-owned memory.
    unsafe { ferrule_cutlass_provider_manifest() }
}

unsafe extern "C" {
    fn ferrule_cutlass_provider_manifest() -> CutlassProviderManifest;
}

/// Discover providers and compile a CUDA model plan from semantic requirements.
pub fn compile_cuda_model_plan(
    requirements: &[LayerKernelRequirements],
) -> Result<ModelKernelPlan> {
    CudaProviderCatalog::discover()?.compile_model_plan(requirements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_catalog_uses_only_the_native_kernel_mask() {
        let native = CutlassProviderManifest {
            kernel_mask: CutlassKernelId::HyperConnectionProducer.mask()
                | CutlassKernelId::MainProjectNorm.mask()
                | CutlassKernelId::HybridMlaAttention.mask()
                | CutlassKernelId::ProposalHead.mask()
                | CutlassKernelId::GroupedFp4Moe.mask(),
        };
        let manifest = cutlass_execution_manifest(native);

        for operation in [
            KernelOperation::AttentionHcPre,
            KernelOperation::FeedForwardHcPre,
            KernelOperation::GroupedFp4Moe,
            KernelOperation::MainProjectNorm,
            KernelOperation::HybridMlaAttention,
            KernelOperation::ProposalHead,
        ] {
            assert!(manifest.supports(operation, ExecutionMode::Inference));
        }
        assert!(!manifest.supports(KernelOperation::MlaOutput, ExecutionMode::Inference));
        assert_eq!(manifest.operations.len(), 6);
    }
}
