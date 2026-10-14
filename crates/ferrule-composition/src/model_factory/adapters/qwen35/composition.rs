//! Qwen3.5 family composition policy for the existing CUDA thread TP×EP factory.
//! Metadata-only queries label support and quality evidence, not GPU readiness.
//! This adapter neither creates owners nor coordinates execution: capacity,
//! checkpoint freshness, completion and Unknown custody stay with the factory.

use super::{NumericFp8Profile, PipelineRankBackend, qwen35};
use ferrule_common::{Error, ParallelismPlan, Result};
use ferrule_model::ModelExecutionBackend;
use ferrule_model::models::qwen35::Qwen35Profile;
use ferrule_model::{ModelFamily, execution::ExecutionPrecisionPolicy};
use ferrule_runtime::parallel::hybrid_topology::{HybridExpertTopology, HybridTopologyError};

/// Declared execution profile, not a checkpoint-binding or GPU-readiness proof.
/// The factory still validates full-depth strict metadata, source freshness,
/// prewarm, device placement/capacity, precision boundaries and startup custody.
/// Queries work in CPU-only builds; `Ok` does not imply CUDA was compiled in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridExecutionProfile {
    pub model: Qwen35Profile,
    pub backend: ModelExecutionBackend,
    pub rank_backend: PipelineRankBackend,
    pub parallelism: ParallelismPlan,
    pub numeric: NumericFp8Profile,
}

/// Factory support is not interchangeable with numerical acceptance evidence.
/// Evidence refers to the documented full35B snapshot, not arbitrary checkpoints,
/// devices, contexts or placements accepted by a metadata-only query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridExecutionCapability {
    /// F32 PP1/2 TP2 EP2 has documented full-model strict GPU evidence.
    F32WithFullModelStrictEvidence,
    /// F32 TP4 is implemented/admitted, but has no full-model strict evidence.
    F32FactoryOnly,
    /// Explicit experimental factory opt-in; production quality is unaccepted.
    W8A16QualityUnaccepted,
}

/// Typed capability rejection, preserved inside `Error::ModelSource`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridCapabilityError {
    ModelProfile,
    Backend,
    RankBackend,
    Parallelism,
    KvOwners,
    ExpertOwners,
    MissingKvPlane,
    W8A16QualityUnaccepted,
}
impl std::fmt::Display for HybridCapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ModelProfile => "hybrid TPxEP requires the strict text-only Qwen3.5-35B numeric FP8 profile",
            Self::Backend => "hybrid TPxEP production factory requires CUDA; no CPU fallback",
            Self::RankBackend => "CUDA hybrid process TP is unsupported; thread owners are required",
            Self::Parallelism => "hybrid TPxEP requires PP1/2 TP2/4 EP2 DP/SP/CP1",
            Self::KvOwners => "hybrid TPxEP requires contiguous PP-major TP KV owners in replica zero",
            Self::ExpertOwners => "hybrid TPxEP requires stage-ordered external EP2 owners and the first TP lane as source",
            Self::MissingKvPlane => "each CUDA hybrid stage requires a full-attention KV plane",
            Self::W8A16QualityUnaccepted => "W8A16 is experimental opt-in only; production quality is unaccepted",
        })
    }
}
impl std::error::Error for HybridCapabilityError {}
impl From<HybridCapabilityError> for Error {
    fn from(source: HybridCapabilityError) -> Self {
        Self::ModelSource {
            source: Box::new(source),
        }
    }
}
impl From<HybridTopologyError> for HybridCapabilityError {
    fn from(source: HybridTopologyError) -> Self {
        match source {
            HybridTopologyError::Parallelism => Self::Parallelism,
            HybridTopologyError::KvOwners => Self::KvOwners,
            HybridTopologyError::ExpertOwners => Self::ExpertOwners,
            HybridTopologyError::MissingKvPlane => Self::MissingKvPlane,
        }
    }
}

/// Family-specific capability over an already validated, model-neutral topology.
/// Available without CUDA; a successful query is not an execution permit.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35CompositionAdapter<'a> {
    topology: &'a HybridExpertTopology,
}

impl<'a> Qwen35CompositionAdapter<'a> {
    pub const fn new(topology: &'a HybridExpertTopology) -> Self {
        Self { topology }
    }

    /// Query the CUDA TP×EP factory, including its explicit W8A16 experimental
    /// opt-in. TP1/EP4/EP8 belong to other factories and are not admitted here.
    pub fn execution_capability(
        &self,
        profile: HybridExecutionProfile,
    ) -> Result<HybridExecutionCapability> {
        use HybridCapabilityError as Rejection;
        if profile.model != Qwen35Profile::Moe35BA3Bfp8
            || self.topology.has_attachments()
            || self.topology.stages().iter().any(|stage| {
                stage
                    .layers()
                    .iter()
                    .any(|layer| layer.routed_experts() == 0)
            })
        {
            return Err(Rejection::ModelProfile.into());
        }
        if profile.backend != ModelExecutionBackend::Cuda {
            return Err(Rejection::Backend.into());
        }
        if profile.rank_backend != PipelineRankBackend::Thread {
            return Err(Rejection::RankBackend.into());
        }
        let p = profile.parallelism;
        if routed_topology(p) != Some(RoutedTopology::TensorRoots) {
            return Err(Rejection::Parallelism.into());
        }
        super::execution_policy::validate_topology(
            p,
            super::execution_policy::OwnerLayout::StageRootsWithExperts,
        )
        .map_err(|_| Rejection::Parallelism)?;
        self.topology
            .validate_external_expert_roster(p)
            .map_err(Rejection::from)?;
        Ok(match profile.numeric {
            NumericFp8Profile::F32 if p.tensor_parallel == 2 => {
                HybridExecutionCapability::F32WithFullModelStrictEvidence
            }
            NumericFp8Profile::F32 => HybridExecutionCapability::F32FactoryOnly,
            NumericFp8Profile::W8A16Bf16 => HybridExecutionCapability::W8A16QualityUnaccepted,
        })
    }

    /// Validate supported production arithmetic/topology, not release readiness.
    /// F32 TP4 is supported but lacks full-model strict evidence. W8A16 is
    /// rejected here even though the factory retains its experimental opt-in.
    ///
    /// ```
    /// use ferrule_common::{ParallelismPlan, Result};
    /// use ferrule_model::{ModelExecutionBackend, models::qwen35::Qwen35Profile};
    /// use ferrule_composition::model_factory::{NumericFp8Profile, PipelineRankBackend};
    /// use ferrule_composition::model_factory::qwen35_composition::{
    ///     HybridExecutionCapability, HybridExecutionProfile, Qwen35CompositionAdapter,
    /// };
    /// use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;
    ///
    /// fn check(layout: &HybridExpertTopology) -> Result<HybridExecutionCapability> {
    ///     let profile = HybridExecutionProfile {
    ///         model: Qwen35Profile::Moe35BA3Bfp8,
    ///         backend: ModelExecutionBackend::Cuda,
    ///         rank_backend: PipelineRankBackend::Thread,
    ///         parallelism: ParallelismPlan {
    ///             pipeline_parallel: 2, tensor_parallel: 2, expert_parallel: 2,
    ///             ..Default::default()
    ///         },
    ///         numeric: NumericFp8Profile::F32,
    ///     };
    ///     let adapter = Qwen35CompositionAdapter::new(layout);
    ///     adapter.validate_production_execution(profile)?;
    ///     adapter.execution_capability(profile)
    /// }
    /// ```
    pub fn validate_production_execution(&self, profile: HybridExecutionProfile) -> Result<()> {
        match self.execution_capability(profile)? {
            HybridExecutionCapability::W8A16QualityUnaccepted => {
                Err(HybridCapabilityError::W8A16QualityUnaccepted.into())
            }
            HybridExecutionCapability::F32WithFullModelStrictEvidence
            | HybridExecutionCapability::F32FactoryOnly => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutedTopology {
    StageExperts,
    TensorRoots,
}

fn routed_topology(p: ParallelismPlan) -> Option<RoutedTopology> {
    match (p.pipeline_parallel, p.tensor_parallel, p.expert_parallel) {
        (2, 1, 2 | 4 | 8) => Some(RoutedTopology::StageExperts),
        (1 | 2, 2 | 4, 2) => Some(RoutedTopology::TensorRoots),
        _ => None,
    }
}

impl super::PipelineBuildOptions {
    /// Compatibility shorthand for strict dense CPU hybrid PP2 precision.
    pub fn hybrid_cpu_precision(
        &self,
        family: &ModelFamily,
    ) -> super::Result<ExecutionPrecisionPolicy> {
        self.hybrid_precision(family, ModelExecutionBackend::Cpu)
    }

    /// Strict dense or numeric routed hybrid policy; binding remains a build-time check.
    pub fn hybrid_precision(
        &self,
        family: &ModelFamily,
        backend: ModelExecutionBackend,
    ) -> super::Result<ExecutionPrecisionPolicy> {
        let moe = *family == ModelFamily::Qwen35Moe;
        let supported_shape = match family {
            ModelFamily::Qwen35 => {
                self.parallelism.pipeline_parallel == 2
                    && self.parallelism.tensor_parallel == 1
                    && self.parallelism.expert_parallel == 1
            }
            ModelFamily::Qwen35Moe => routed_topology(self.parallelism).is_some(),
            _ => false,
        };
        if !supported_shape
            || (moe
                && (backend != ModelExecutionBackend::Cuda
                    || self.rank_backend != PipelineRankBackend::Thread
                    || self.devices.is_none()))
        {
            let message = "hybrid requires dense PP2 TP1 EP1 or Qwen35Moe CUDA thread PP2 TP1 EP2/4/8 or PP1/2 TP2/4 EP2 with explicit root-then-stage-EP devices; MoE process/CPU unsupported";
            return Err(if moe {
                qwen35::ep_unsupported(message)
            } else {
                ferrule_common::Error::ModelSource {
                    source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
                        "hybrid pipeline",
                        message,
                    )),
                }
                .into()
            });
        }
        let policy = self.hybrid_policy(family);
        self.validate_policy(backend, policy.layout, policy.execution, policy.placement)
            .map_err(|e| {
                if moe {
                    qwen35::ep_unsupported(e.to_string())
                } else {
                    e
                }
            })?;
        self.require_compiled_backend(backend).map_err(|e| {
            if moe {
                qwen35::ep_unsupported(e.to_string())
            } else {
                e
            }
        })?;
        Ok(ExecutionPrecisionPolicy::f32())
    }
    pub(super) fn hybrid_policy(
        &self,
        family: &ModelFamily,
    ) -> super::execution_policy::TopologyPolicy {
        use super::execution_policy::*;
        let moe = *family == ModelFamily::Qwen35Moe;
        TopologyPolicy {
            layout: self.owner_layout(),
            execution: ExecutionCapabilities {
                ranks: if moe {
                    RankCapability::ThreadOnly
                } else {
                    RankCapability::ThreadOrProcess
                },
                tensor: if moe {
                    TensorCapability::DeviceThreads { degrees: &[2, 4] }
                } else {
                    TensorCapability::Unsharded
                },
            },
            placement: PlacementPolicy {
                selection: if moe {
                    DeviceSelection::Explicit
                } else {
                    DeviceSelection::AutomaticOrExplicit
                },
                sharing: DeviceSharing::Colocated,
            },
        }
    }
}

pub(super) fn attach_pipeline(
    plan: super::ResidentModelBuildPlan,
    options: super::PipelineBuildOptions,
    execution: super::ResolvedExecution,
) -> super::Result<super::ResidentModelBuildPlan> {
    use super::*;
    options.hybrid_precision(&plan.request.family, plan.backend)?;
    let mut plan = plan.with_resolved_pipeline(options.clone(), execution)?;
    plan.backend_profile = match (plan.backend, options.rank_backend) {
        (ModelExecutionBackend::Cpu, PipelineRankBackend::Thread) => {
            "cpu-hybrid-pipeline-f32-serial"
        }
        (ModelExecutionBackend::Cuda, PipelineRankBackend::Thread) => {
            "cuda-hybrid-pipeline-f32-serial"
        }
        (ModelExecutionBackend::Cpu, PipelineRankBackend::Process) => {
            "cpu-hybrid-pipeline-process-f32-serial"
        }
        (ModelExecutionBackend::Cuda, PipelineRankBackend::Process) => {
            "cuda-hybrid-pipeline-process-f32-serial"
        }
    };
    if plan.request.family == ModelFamily::Qwen35Moe {
        let moe = plan.request.family_options.qwen35().unwrap().moe().unwrap();
        if moe.device_capacity.is_some()
            || moe.host_cache.mode
                != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
        {
            return Err(invalid(
                "Qwen35 PPxEP requires full shared prewarm and all-owned residency; local CUDA cache overrides unsupported",
            ));
        }
        let profile = moe.numeric_profile;
        plan.backend_profile = match (profile, options.parallelism.tensor_parallel > 1) {
            (NumericFp8Profile::F32, true) => "cuda-hybrid-tp-ep-f32-tf32x3-serial",
            (NumericFp8Profile::W8A16Bf16, true) => {
                "cuda-hybrid-tp-ep-w8a16-quality-unaccepted-serial"
            }
            (NumericFp8Profile::F32, false) => "cuda-hybrid-pp2-ep-f32-tf32x3-serial",
            (NumericFp8Profile::W8A16Bf16, false) => {
                "cuda-hybrid-pp2-ep-w8a16-bf16-f32-accumulate-serial"
            }
        };
    }
    if let Some(qwen35::Qwen35ResolvedAdapter::Moe {
        expert_placement, ..
    }) = plan.request.family_options.qwen35_mut()
    {
        *expert_placement = None;
    }
    Ok(plan)
}

/// Catalog lowering freezes the owner constructor and the matching option policy.
/// No checkpoint payload, owner, or device is opened here.
pub(super) fn select_strategy(
    entry: &super::ModelImplementation,
    backend: super::BackendSelection,
    request: &super::SelectionRequest,
    options: &super::ModelFactoryOptions,
) -> super::Result<super::ResolvedStrategy> {
    use super::*;
    let moe = entry.family == ModelFamily::Qwen35Moe;
    if request.numeric_model.is_some() {
        if !moe {
            return Err(invalid(
                "numeric BF16 requires the exact Qwen3.5 MoE FP8 family",
            ));
        }
        return bf16_serving::select_numeric(backend, request, options);
    }
    request.reject_specialized()?;
    let parallel = request.serving.parallel.as_ref();
    let backend = Option::<ModelExecutionBackend>::from(backend).unwrap_or(if moe {
        ModelExecutionBackend::Cuda
    } else {
        ModelExecutionBackend::Cpu
    });
    let (construction, policy, constraints) = match parallel {
        Some(options)
            if moe
                && options.parallelism.pipeline_parallel == 1
                && options.parallelism.tensor_parallel == 1 =>
        {
            Qwen35ExpertPlacement::from_options(options, backend.into())?;
            (
                ResolvedConstruction::Resident(qwen35::build_experts),
                &EXPERT_OPTIONS,
                "Qwen35Moe CUDA thread/process EP2/4/8; explicit distinct devices; resident root shares expert0 card; root-only KV; full owned prewarm; no TP/PP/restart",
            )
        }
        Some(options) => {
            options.hybrid_precision(&entry.family, backend)?;
            (
                ResolvedConstruction::Pipeline(build_hybrid_pipeline),
                if moe {
                    &MOE_PIPELINE_OPTIONS
                } else {
                    &DENSE_PIPELINE_OPTIONS
                },
                if moe {
                    "strict Qwen3.5-35B numeric FP8 CUDA thread PP2 TP1 EP2/4/8 or PP1/2 TP2/4 EP2; PP-major TP roots then stage EP children; aggregate colocation admission; full shared prewarm; F32 default, W8A16 opt-in quality-unaccepted; F32 boundaries; no process/restart"
                } else {
                    "strict dense Qwen3.5-0.8B CPU/CUDA F32 thread/process PP2 TP1 EP1; compact local KV, logical L/physical 2L; bounded full-vocabulary IPC; no MoE PPxEP/TPxEP/restart; strict binding and aggregate live admission required"
                },
            )
        }
        None => (
            ResolvedConstruction::Resident(entry.build),
            &self::RESIDENT_OPTIONS,
            "full-depth Qwen3.5; batch <= min(32, context); no prefix cache/proposals; profile binding and source identity not yet validated",
        ),
    };
    let mut execution = ResolvedExecution::new(
        backend,
        construction,
        if moe {
            StorageEncoding::NumericFp8E4m3fnBf16Scales
        } else {
            StorageEncoding::ProfileDefined
        },
        if moe {
            match options.numeric_fp8_profile.unwrap_or_default() {
                NumericFp8Profile::F32 => ArithmeticPrecision::F32Tf32x3,
                NumericFp8Profile::W8A16Bf16 => ArithmeticPrecision::W8A16Bf16F32Accumulate,
            }
        } else {
            ArithmeticPrecision::F32
        },
        constraints,
    );
    execution.topology_policy = parallel.map(|options| match construction {
        ResolvedConstruction::Resident(_) => qwen35::EXPERT_TOPOLOGY,
        ResolvedConstruction::Pipeline(_) => options.hybrid_policy(&entry.family),
    });
    execution.numeric_profile = moe.then(|| options.numeric_fp8_profile.unwrap_or_default());
    execution.adjustment_reason = qwen35::ADJUSTMENT_REASON;
    Ok(ResolvedStrategy {
        execution,
        backend,
        catalog_backend: backend.into(),
        resolve: entry.resolve,
        policy,
        parallel: parallel.cloned(),
    })
}

use super::ModelOptionPolicy;

const RESIDENT_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: resident_options,
    parallel: super::reject_parallel,
    adapt: qwen35::adapt_options,
    finish: qwen35::finish_options,
};
const DENSE_PIPELINE_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: dense_pipeline_options,
    parallel: attach_pipeline,
    adapt: qwen35::adapt_options,
    finish: qwen35::finish_options,
};
const MOE_PIPELINE_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: moe_pipeline_options,
    parallel: attach_pipeline,
    adapt: qwen35::adapt_options,
    finish: qwen35::finish_options,
};
const EXPERT_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: resident_options,
    parallel: qwen35::attach_expert_options,
    adapt: qwen35::adapt_options,
    finish: qwen35::finish_options,
};

pub(super) fn resident_options(
    family: &ModelFamily,
    serving: &super::ServingModelOptions,
    options: super::ModelFactoryOptions,
    construction: super::ResolvedConstruction,
) -> super::Result<super::ModelFactoryOptions> {
    let mut options = serving.resolve_options(construction, options)?;
    qwen35::resolve_serving_options(family, serving, &mut options)?;
    Ok(options)
}

fn dense_pipeline_options(
    family: &ModelFamily,
    serving: &super::ServingModelOptions,
    options: super::ModelFactoryOptions,
    construction: super::ResolvedConstruction,
) -> super::Result<super::ModelFactoryOptions> {
    if options.qwen35_host_cache.is_some() {
        return Err(super::invalid(
            "pipeline serving does not implement expert prewarm",
        ));
    }
    resident_options(family, serving, options, construction)
}

pub(super) fn moe_pipeline_options(
    family: &ModelFamily,
    serving: &super::ServingModelOptions,
    mut options: super::ModelFactoryOptions,
    _: super::ResolvedConstruction,
) -> super::Result<super::ModelFactoryOptions> {
    // Routed hybrid pipelines consume explicit full-prewarm budgets, unlike
    // standard/dense pipelines. Family adaptation consumes those explicit values.
    super::selection::validate_serving_engine(
        serving,
        super::ResolvedConstruction::Pipeline(super::build_hybrid_pipeline),
    )?;
    super::selection::apply_common_serving_options(serving, &mut options);
    qwen35::resolve_serving_options(family, serving, &mut options)?;
    Ok(options)
}
