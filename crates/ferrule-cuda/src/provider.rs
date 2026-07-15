//! CUDA kernel-provider discovery and executable-plan compilation.
//!
//! Provider selection happens during model preparation. The hot path consumes
//! resolved [`LaunchDescriptor`] values and never queries architecture strings,
//! environment variables, or dynamic trait objects.

use ferrule_common::{Error, Result};

pub use ferrule_common::kernel_plan::{
    KernelId, KernelPhase, KernelProviderId, LaunchDescriptor, LayerKernelPlan, LayerKernelPlanSet,
    ModelKernelPlan, ProviderManifest, ProviderRegistry, RowBucket, WeightBinding, WeightLayout,
};

/// Providers and native catalogs available in this build.
#[derive(Debug, Clone)]
pub struct CudaProviderCatalog {
    registry: ProviderRegistry,
    cutlass: Option<crate::cutlass::CutlassProviderManifest>,
}

impl CudaProviderCatalog {
    /// Discover compiled providers once during prepare/compile.
    pub fn discover() -> Result<Self> {
        let mut registry = ProviderRegistry::with_cuda_oxide();
        let cutlass = crate::cutlass::provider_manifest();
        if let Some(manifest) = cutlass {
            if manifest.abi_version != crate::cutlass::CUTLASS_ABI_VERSION {
                return Err(Error::Internal(format!(
                    "CUTLASS provider ABI mismatch: native={} rust={}",
                    manifest.abi_version,
                    crate::cutlass::CUTLASS_ABI_VERSION
                )));
            }
            let execution_manifest =
                crate::cutlass::execution_provider_manifest().ok_or_else(|| {
                    Error::Internal("CUTLASS provider manifest conversion failed".into())
                })?;
            registry.register(execution_manifest);
        }
        Ok(Self { registry, cutlass })
    }

    pub const fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    pub const fn cutlass_manifest(&self) -> Option<crate::cutlass::CutlassProviderManifest> {
        self.cutlass
    }

    /// Compile the currently implemented DSV4 semantic phases for every row
    /// bucket. Unsupported or absent phases remain on the eager fallback.
    pub fn compile_model_plan(
        &self,
        layer_count: usize,
        hidden_size: usize,
    ) -> Result<ModelKernelPlan> {
        let mut model_plan = ModelKernelPlan::new(layer_count);
        let Some(cutlass) = self.cutlass else {
            return Ok(model_plan);
        };
        if !cutlass.supports(crate::cutlass::CutlassKernelId::Bf16MmaSync) {
            return Ok(model_plan);
        }

        for layer in 0..layer_count {
            let plan_set = model_plan
                .layer_mut(layer)
                .ok_or_else(|| Error::Internal(format!("kernel plan lost layer slot {layer}")))?;
            for rows in RowBucket::ALL {
                let workspace_len = rows
                    .rows()
                    .checked_mul(hidden_size)
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "CUTLASS BF16 workspace overflow: rows={} hidden={hidden_size}",
                            rows.rows()
                        ))
                    })?;
                let kernel = KernelId::new(
                    KernelProviderId::CutlassCubin,
                    KernelPhase::CompressorProjection,
                    rows,
                )
                .with_variant(crate::cutlass::CutlassKernelId::Bf16MmaSync as u8);
                let descriptor = LaunchDescriptor::new(kernel, (0, 0, 0), (0, 0, 0))
                    .provider_managed()
                    .capture_safe();
                let mut layer_plan = LayerKernelPlan::new(rows);
                layer_plan.workspace_len = workspace_len;
                layer_plan.set_phase(descriptor);
                plan_set.set_plan(layer_plan);
            }
        }
        Ok(model_plan)
    }
}

/// Discover providers and compile the default CUDA model plan.
pub fn compile_cuda_model_plan(layer_count: usize, hidden_size: usize) -> Result<ModelKernelPlan> {
    CudaProviderCatalog::discover()?.compile_model_plan(layer_count, hidden_size)
}
