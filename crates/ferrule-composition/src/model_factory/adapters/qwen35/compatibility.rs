//! Legacy public facades. Provider calculations and transport data are shared.
use super::*;
use crate::model_factory::parallel::{
    budget::{ExpertPhysicalPlan, ExpertRootMemory, PhysicalCardCapacity, PhysicalCardMemory},
    inputs::{ExpertPlacementConfig, NumericCapacityLimits},
};
fn legacy_error(error: Error) -> Error {
    if let Error::Backend {
        source: ferrule_common::Error::ModelSource { source },
    } = &error
        && let Some(unsupported) =
            source.downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
        && unsupported.operator == "numeric_expert_provider"
    {
        return ep_unsupported(unsupported.reason.clone());
    }
    error
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ExpertPlacement(pub(in crate::model_factory) ExpertPlacementConfig);
impl std::ops::Deref for Qwen35ExpertPlacement {
    type Target = ExpertPlacementConfig;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Qwen35ExpertPlacement {
    pub fn rank_backend(&self) -> PipelineRankBackend {
        self.0.rank_backend()
    }
    pub fn root_device(&self) -> usize {
        self.0.root_device()
    }
    pub fn expert_devices(&self) -> &[usize] {
        self.0.expert_devices()
    }
    pub fn expert_parallel(&self) -> usize {
        self.0.expert_parallel()
    }
    pub fn new(degree: usize, devices: Vec<usize>) -> Result<Self> {
        validate_degree(degree)?;
        ExpertPlacementConfig::new(degree, devices)
            .map(Self)
            .map_err(legacy_error)
    }
    pub(in crate::model_factory) fn from_options(
        options: &PipelineBuildOptions,
        backend: BackendSelection,
    ) -> Result<Self> {
        validate_degree(options.parallelism.expert_parallel)?;
        ExpertPlacementConfig::from_options(options, backend)
            .map(Self)
            .map_err(legacy_error)
    }
    pub fn validate_expert_shape(&self, layers: usize, experts: usize) -> Result<usize> {
        self.0
            .validate_expert_shape(layers, experts)
            .map_err(legacy_error)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35ExpertRootMemory {
    pub weights_bytes: usize,
    pub state_bytes: usize,
    pub kv_bytes: usize,
    pub workspace_bytes: usize,
}
impl Qwen35ExpertRootMemory {
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        let root =
            ExpertRootMemory::for_resources(resources, options, kv_pages).map_err(legacy_error)?;
        Ok(Self {
            weights_bytes: root.weights_bytes,
            state_bytes: root.state_bytes,
            kv_bytes: root.kv_bytes,
            workspace_bytes: root.workspace_bytes,
        })
    }
}
impl From<Qwen35ExpertRootMemory> for ExpertRootMemory {
    fn from(root: Qwen35ExpertRootMemory) -> Self {
        Self {
            weights_bytes: root.weights_bytes,
            state_bytes: root.state_bytes,
            kv_bytes: root.kv_bytes,
            workspace_bytes: root.workspace_bytes,
        }
    }
}
pub type Qwen35PhysicalCardMemory = PhysicalCardMemory;
pub type Qwen35PhysicalCardCapacity = PhysicalCardCapacity;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ExpertPhysicalPlan(ExpertPhysicalPlan);
impl Qwen35ExpertPhysicalPlan {
    pub fn new(
        placement: &Qwen35ExpertPlacement,
        root: Qwen35ExpertRootMemory,
        weights: &[usize],
        workspace: usize,
        margin: usize,
    ) -> Result<Self> {
        ExpertPhysicalPlan::new(&placement.0, root.into(), weights, workspace, margin)
            .map(Self)
            .map_err(legacy_error)
    }
    pub fn cards(&self) -> &[Qwen35PhysicalCardMemory] {
        self.0.cards()
    }
    pub fn check_capacity(&self, capacities: &[Qwen35PhysicalCardCapacity]) -> Result<()> {
        self.0.check_capacity(capacities).map_err(legacy_error)
    }
}
/// Explicit capacity inputs for metadata admission and single-GPU construction.
/// Cache bytes include resident + pending compressed experts and the shared scratch.
/// Host residency is separately configured by HostExpertCacheOptions; no pinned hotset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35MoeCapacityLimits {
    pub max_experts: usize,
    pub max_bytes: usize,
    pub scratch_bytes: usize,
    /// F32 operator workspace excluding numeric scratch (already in max_bytes).
    pub workspace_bytes: usize,
    pub kv_bytes: usize,
    pub device_bytes: usize,
    /// Separate allowance for allocator segments and library/context overhead.
    pub allocator_margin_bytes: usize,
    pub max_positions: usize,
    pub max_sequences: usize,
    pub max_batch_tokens: usize,
}

impl Default for Qwen35MoeCapacityLimits {
    fn default() -> Self {
        Self {
            max_experts: 1024,
            max_bytes: 4usize << 30,
            scratch_bytes: 64 << 20,
            workspace_bytes: 1 << 30,
            kv_bytes: 1 << 30,
            device_bytes: 16usize << 30,
            allocator_margin_bytes: 512 << 20,
            max_positions: 1024,
            max_sequences: 4,
            max_batch_tokens: 32,
        }
    }
}

impl Qwen35MoeCapacityLimits {
    /// The explicit GPU policy, not the unrelated pageable/pinned host options.
    /// The model options API accepts only these bounded limits, never KeepAll.
    pub fn expert_cache_policy(&self) -> Result<ferrule_model::transformer::ExpertCachePolicy> {
        NumericCapacityLimits::from(*self)
            .expert_cache_policy()
            .map_err(|_| {
                invalid("35B requires nonzero bounded expert cache with room beyond scratch")
            })
    }
}

impl From<Qwen35MoeCapacityLimits> for NumericCapacityLimits {
    fn from(value: Qwen35MoeCapacityLimits) -> Self {
        Self {
            max_experts: value.max_experts,
            max_bytes: value.max_bytes,
            scratch_bytes: value.scratch_bytes,
            workspace_bytes: value.workspace_bytes,
            kv_bytes: value.kv_bytes,
            device_bytes: value.device_bytes,
            allocator_margin_bytes: value.allocator_margin_bytes,
            max_positions: value.max_positions,
            max_sequences: value.max_sequences,
            max_batch_tokens: value.max_batch_tokens,
        }
    }
}

fn validate_degree(degree: usize) -> Result<()> {
    if !matches!(degree, 2 | 4 | 8) {
        return Err(ep_unsupported("Qwen3.5 EP requires degree 2, 4 or 8"));
    }
    Ok(())
}
