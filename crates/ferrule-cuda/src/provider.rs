//! CUDA kernel-provider discovery and executable-plan compilation.
//!
//! Provider selection happens during model preparation. The hot path consumes
//! resolved [`LaunchDescriptor`] values and never queries architecture strings,
//! environment variables, or dynamic trait objects.

use ferrule_common::{Error, Result};

pub use ferrule_common::kernel_plan::{
    KernelId, KernelOperation, KernelPhase, KernelProviderId, LaunchDescriptor, LayerKernelPlan,
    LayerKernelRequirements, LinearBundleRequirement, ModelKernelPlan, ProviderManifest,
    ProviderRegistry, WeightBinding, WeightLayout,
};

/// Providers and native catalogs available in this build.
#[derive(Debug, Clone)]
pub struct CudaProviderCatalog {
    registry: ProviderRegistry,
    cutlass: crate::cutlass::CutlassProvider,
}

impl CudaProviderCatalog {
    /// Discover compiled providers once during prepare/compile.
    pub fn discover() -> Result<Self> {
        let mut registry = ProviderRegistry::with_cuda_oxide();
        let cutlass = crate::cutlass::discover_provider()?;
        registry.register(cutlass.execution_manifest()?);
        Ok(Self { registry, cutlass })
    }

    pub const fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    pub const fn cutlass_manifest(&self) -> crate::cutlass::CutlassProviderManifest {
        self.cutlass.manifest()
    }

    /// Compile provider-neutral requirements into one semantic plan per layer.
    /// Native providers own row-count schedule dispatch. Missing capabilities
    /// are fatal; production has no fallback path.
    pub fn compile_model_plan(
        &self,
        requirements: &[LayerKernelRequirements],
    ) -> Result<ModelKernelPlan> {
        let mut model_plan = ModelKernelPlan::new(requirements.len());
        let cutlass = self.cutlass;

        for (layer, requirements) in requirements.iter().enumerate() {
            let layer_plan = model_plan
                .layer_mut(layer)
                .ok_or_else(|| Error::Internal(format!("kernel plan lost layer slot {layer}")))?;
            for requirement in &requirements.linear_bundles {
                require_cutlass_bundle(cutlass, requirement)?;
                set_provider_operation(layer_plan, requirement.operation);
            }
            for &operation in &requirements.semantic_operations {
                require_semantic_operation(cutlass, operation)?;
                set_provider_operation(layer_plan, operation);
            }
        }
        Ok(model_plan)
    }
}

fn set_provider_operation(plan: &mut LayerKernelPlan, operation: KernelOperation) {
    let kernel = KernelId::new(KernelProviderId::CutlassCubin, operation);
    let descriptor = LaunchDescriptor::new(kernel, (0, 0, 0), (0, 0, 0))
        .provider_managed()
        .capture_safe();
    plan.set_operation(descriptor);
}

fn require_semantic_operation(
    provider: crate::cutlass::CutlassProvider,
    operation: KernelOperation,
) -> Result<()> {
    let kernel = match operation {
        KernelOperation::AttentionHcPre | KernelOperation::FeedForwardHcPre => {
            crate::cutlass::CutlassKernelId::HcProducerSm121
        }
        KernelOperation::MlaOutput => crate::cutlass::CutlassKernelId::MlaOutputSm121,
        KernelOperation::SharedFfn => crate::cutlass::CutlassKernelId::SharedFfnSm121,
        KernelOperation::RoutedFp4Moe => crate::cutlass::CutlassKernelId::StableFrameFp4MoeSm121,
        KernelOperation::DsparkMainProjectNorm => {
            crate::cutlass::CutlassKernelId::DsparkMainProjectNormSm121
        }
        KernelOperation::DsparkHybridMlaAttention => {
            crate::cutlass::CutlassKernelId::DsparkHybridMlaAttentionSm121
        }
        KernelOperation::DsparkProposalHead => {
            crate::cutlass::CutlassKernelId::DsparkProposalHeadSm121
        }
        _ => {
            return Err(Error::Internal(format!(
                "no SM121 semantic provider binding for operation={operation:?}"
            )));
        }
    };
    require_kernel(provider, operation, kernel)
}

fn require_cutlass_bundle(
    provider: crate::cutlass::CutlassProvider,
    requirement: &LinearBundleRequirement,
) -> Result<()> {
    let outputs_valid = requirement.output_features.len() == 2
        && requirement
            .output_features
            .iter()
            .all(|features| *features > 0);
    if !outputs_valid {
        return Err(Error::Internal(format!(
            "SM121 operation {:?} requires exactly two non-empty outputs",
            requirement.operation
        )));
    }
    match (requirement.operation, requirement.weight_layout) {
        (KernelOperation::MlaQueryAKv, WeightLayout::Fp8E4m3BlockScaled) => {
            if !requirement.input_features.is_multiple_of(128) {
                return Err(Error::Internal(format!(
                    "SM121 FP8 QueryA+KV requires K128, got {}",
                    requirement.input_features
                )));
            }
            require_kernel(
                provider,
                requirement.operation,
                crate::cutlass::CutlassKernelId::Fp8QueryAKvSm121,
            )
        }
        (
            KernelOperation::MainCompressorProjection
            | KernelOperation::IndexerCompressorProjection,
            WeightLayout::Bf16RowMajor,
        ) => {
            if !requirement.input_features.is_multiple_of(8)
                || !requirement
                    .output_features
                    .iter()
                    .all(|features| features.is_multiple_of(4))
            {
                return Err(Error::Internal(format!(
                    "SM121 BF16 compressor shape is unsupported: operation={:?} K={} N={:?}",
                    requirement.operation, requirement.input_features, requirement.output_features
                )));
            }
            require_kernel(
                provider,
                requirement.operation,
                crate::cutlass::CutlassKernelId::Bf16CompressorSm121,
            )
        }
        _ => Err(Error::Internal(format!(
            "no SM121 provider binding for operation={:?} layout={:?}",
            requirement.operation, requirement.weight_layout
        ))),
    }
}

fn require_kernel(
    provider: crate::cutlass::CutlassProvider,
    operation: KernelOperation,
    kernel: crate::cutlass::CutlassKernelId,
) -> Result<()> {
    if !provider.supports(kernel) {
        return Err(Error::Internal(format!(
            "required SM121 {operation:?} superkernel is not published"
        )));
    }
    Ok(())
}

/// Discover providers and compile a CUDA model plan from semantic requirements.
pub fn compile_cuda_model_plan(
    requirements: &[LayerKernelRequirements],
) -> Result<ModelKernelPlan> {
    CudaProviderCatalog::discover()?.compile_model_plan(requirements)
}
