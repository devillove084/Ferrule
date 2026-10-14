//! Provider inputs lowered by a family adapter before shared planning.
use super::plan::ParallelExecutionContract;
use super::provider_unsupported;
use crate::Result;
use crate::model_factory::{
    BackendSelection, NumericFp8Profile, PipelineBuildOptions, PipelineRankBackend,
    execution_policy,
};
use ferrule_model::transformer::host_experts::HostExpertCacheOptions;
use ferrule_model::{ModelExecutionBackend, ModelFamily};

#[derive(Debug, Clone)]
pub(in crate::model_factory) struct NumericProviderInputs {
    pub(in crate::model_factory) model_label: &'static str,
    // Runner provenance only: provider selection never matches on this identity.
    pub(in crate::model_factory) family: ModelFamily,
    pub(in crate::model_factory) limits: NumericCapacityLimits,
    pub(in crate::model_factory) numeric_profile: NumericFp8Profile,
    /// Sealed model-owned rank boundary; execution still requires the model validator.
    pub(in crate::model_factory) parallel_execution: ParallelExecutionContract,
    pub(in crate::model_factory) host_cache: HostExpertCacheOptions,
    pub(in crate::model_factory) placement: Option<ExpertPlacementConfig>,
}

pub(in crate::model_factory) const EXPERT_TOPOLOGY: execution_policy::TopologyPolicy =
    execution_policy::TopologyPolicy {
        layout: execution_policy::OwnerLayout::RootWithExperts,
        execution: execution_policy::ExecutionCapabilities {
            ranks: execution_policy::RankCapability::ThreadOrProcess,
            tensor: execution_policy::TensorCapability::Unsharded,
        },
        placement: execution_policy::PlacementPolicy {
            selection: execution_policy::DeviceSelection::Explicit,
            sharing: execution_policy::DeviceSharing::Distinct,
        },
    };

/// Explicit expert placement, separate from tensor and KV rank topology.
/// The root and expert owner zero use separate contexts on devices[0].
/// No placement is inferred from the number of visible GPUs.
#[derive(Debug, Clone)]
pub struct ExpertPlacementConfig {
    pub(in crate::model_factory) devices: Vec<usize>,
    pub(in crate::model_factory) rank_backend: PipelineRankBackend,
    pub(in crate::model_factory) rank_timeout: std::time::Duration,
    #[cfg(unix)]
    pub(in crate::model_factory) process_launch:
        Option<ferrule_runtime::parallel::process::ProcessLaunch>,
}

impl PartialEq for ExpertPlacementConfig {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(unix)]
        let launch_equal = match (&self.process_launch, &other.process_launch) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.executable == b.executable && a.args == b.args && a.environment == b.environment
            }
            _ => false,
        };
        #[cfg(not(unix))]
        let launch_equal = true;
        self.devices == other.devices
            && self.rank_backend == other.rank_backend
            && self.rank_timeout == other.rank_timeout
            && launch_equal
    }
}
impl Eq for ExpertPlacementConfig {}

impl ExpertPlacementConfig {
    pub fn new(expert_parallel: usize, devices: Vec<usize>) -> Result<Self> {
        if expert_parallel < 2 {
            return Err(provider_unsupported(
                "external experts require at least two owners",
            ));
        }
        let topology = execution_policy::validate_topology(
            ferrule_common::ParallelismPlan {
                expert_parallel,
                ..Default::default()
            },
            EXPERT_TOPOLOGY.layout,
        )
        .map_err(|e| provider_unsupported(e.to_string()))?;
        execution_policy::validate_placement(topology, execution_policy::PlacementTarget::Device, Some(&devices), EXPERT_TOPOLOGY.placement)
            .map_err(|e| provider_unsupported(format!("numeric EP{expert_parallel} requires exactly {expert_parallel} distinct CUDA ordinals; root = devices[0], experts = the same list: {e}")))?;
        Ok(Self {
            devices,
            rank_backend: PipelineRankBackend::Thread,
            rank_timeout: std::time::Duration::from_secs(30),
            #[cfg(unix)]
            process_launch: None,
        })
    }

    pub fn rank_backend(&self) -> PipelineRankBackend {
        self.rank_backend
    }

    pub fn root_device(&self) -> usize {
        self.devices[0]
    }

    pub fn expert_devices(&self) -> &[usize] {
        &self.devices
    }

    pub fn expert_parallel(&self) -> usize {
        self.devices.len()
    }

    pub fn validate_expert_shape(&self, layers: usize, experts_per_layer: usize) -> Result<usize> {
        if layers == 0
            || experts_per_layer == 0
            || !experts_per_layer.is_multiple_of(self.expert_parallel())
        {
            return Err(provider_unsupported(
                "numeric EP requires a nonempty expert shape divisible by the owner count",
            ));
        }
        layers
            .checked_mul(experts_per_layer / self.expert_parallel())
            .ok_or_else(|| provider_unsupported("numeric owned expert count overflow"))
    }

    pub(in crate::model_factory) fn from_options(
        options: &PipelineBuildOptions,
        backend: BackendSelection,
    ) -> Result<Self> {
        if backend == BackendSelection::Cpu {
            return Err(provider_unsupported(
                "numeric EP requires CUDA; CPU is unsupported",
            ));
        }
        options
            .validate_policy(
                ModelExecutionBackend::Cuda,
                EXPERT_TOPOLOGY.layout,
                EXPERT_TOPOLOGY.execution,
                EXPERT_TOPOLOGY.placement,
            )
            .map_err(|e| provider_unsupported(e.to_string()))?;
        let p = options.parallelism;
        let mut placement = Self::new(
            p.expert_parallel,
            options.devices.clone().ok_or_else(|| {
                provider_unsupported(
                    "numeric EP requires explicit --devices; no automatic multi-GPU allocation",
                )
            })?,
        )?;
        placement.rank_backend = options.rank_backend;
        placement.rank_timeout = options.rank_timeout;
        #[cfg(unix)]
        {
            placement.process_launch = options.process_launch.clone();
        }
        Ok(placement)
    }
}

/// Explicit capacity inputs for metadata admission and single-GPU construction.
/// Cache bytes include resident + pending compressed experts and the shared scratch.
/// Host residency is separately configured by HostExpertCacheOptions; no pinned hotset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericCapacityLimits {
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

impl Default for NumericCapacityLimits {
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

impl NumericCapacityLimits {
    /// The explicit GPU policy, not the unrelated pageable/pinned host options.
    /// The model options API accepts only these bounded limits, never KeepAll.
    pub fn expert_cache_policy(&self) -> Result<ferrule_model::transformer::ExpertCachePolicy> {
        use ferrule_model::transformer::{ExpertCacheLimits, ExpertCachePolicy};
        if self.max_experts == 0 || self.scratch_bytes == 0 || self.max_bytes <= self.scratch_bytes
        {
            return Err(crate::model_factory::invalid(
                "numeric provider requires nonzero bounded expert cache with room beyond scratch",
            ));
        }
        Ok(ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: self.max_experts,
            max_bytes: self.max_bytes,
        }))
    }
}

#[cfg(unix)]
pub(crate) fn process_owner_config(
    timeout: std::time::Duration,
) -> ferrule_runtime::parallel::process::ProcessOwnerConfig {
    ferrule_runtime::parallel::process::ProcessOwnerConfig {
        startup_timeout: timeout,
        command_timeout: timeout,
        ..Default::default()
    }
}
