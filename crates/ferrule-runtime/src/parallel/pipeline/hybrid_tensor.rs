//! Explicit CPU-reference hybrid PP x TP with optional, external EP owners.
//! Only construction and owner-local model binding live here. Forward, commit,
//! cancellation and physical custody use the existing pipeline and worker.
use super::tensor_scope::TensorScopedStateProgram;
use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_model::ModelExecutionBackend;
mod admission;
use super::*;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::expert::{ExpertOwnerStats, ExpertRankWorker};
use crate::parallel::hybrid_topology::HybridExpertTopology;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
use admission::{reject_cpu_numeric_fp8, seal_binding};
use ferrule_common::{ParallelGroupId, execution::KvLayoutSchema};
use ferrule_model::decoder::{
    CpuKvView, HybridLayerSchema, HybridStateSchema, PackedDecoderBatch, TypedCpuPagedKvPool,
};
use ferrule_model::transformer::hybrid_parallel::{
    HybridParallelPlan, HybridTensorCpuSegment, HybridTensorSequenceLifecycle,
    HybridTensorSequenceState,
};
use ferrule_model::transformer::{DecoderModelSpec, ExpertDispatchLimits, FeedForward};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct HybridCpuTensorConfig {
    pub collective_limits: HostCollectiveLimits,
    pub collective_timeout: Duration,
    pub expert_limits: ExpertDispatchLimits,
    pub expert_timeout: Duration,
    /// Resident model weights + session/working recurrent state + physical KV.
    /// Temporary scalar-CPU workspace and mapped checkpoint bytes are additional.
    pub max_owner_resident_bytes: usize,
    pub max_expert_weight_bytes: usize,
}
impl HybridCpuTensorConfig {
    fn validate(&self) -> Result<()> {
        if self.expert_timeout.is_zero()
            || std::time::Instant::now()
                .checked_add(self.expert_timeout)
                .is_none()
            || self.max_owner_resident_bytes == 0
            || self.max_expert_weight_bytes == 0
            || self.expert_limits.max_tokens == 0
            || self.expert_limits.max_bytes == 0
        {
            return Err(error("invalid hybrid CPU resource/dispatch limits"));
        }
        Ok(())
    }
}

pub use super::tensor_scope::translate_hybrid_schedule;

mod program;
pub use program::HybridCpuTensorPipelineStageProgram;

pub type HybridCpuTensorStage = PipelineStage<
    PagedKvBackend<TypedCpuPagedKvPool<HybridTensorSequenceState>>,
    TensorScopedStateProgram<HybridCpuTensorPipelineStageProgram>,
>;

fn geometry(schema: &HybridStateSchema) -> Result<(usize, usize)> {
    schema
        .layers()
        .iter()
        .find_map(|layer| match layer {
            HybridLayerSchema::FullAttention {
                kv_heads, head_dim, ..
            } => Some((*kv_heads, *head_dim)),
            _ => None,
        })
        .ok_or_else(|| error("hybrid CPU stage needs a full-attention KV plane"))
}
fn description(
    spec: &DecoderModelSpec,
    plan: LayerSegmentPlan,
    schema: &HybridStateSchema,
    config: PipelineConfig,
    group: Option<ExpertGroup>,
) -> Result<PipelineStageDescription> {
    let (kv_heads, head_dim) = geometry(schema)?;
    Ok(PipelineStageDescription {
        plan,
        config,
        physical_pages: config
            .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))?,
        hidden: spec.hidden_size(),
        vocabulary: spec.vocab_size(),
        kv_heads,
        head_dim,
        expert_group: group,
    })
}
impl HybridCpuTensorStage {
    /// Construct and bind on the physical owner. Fallible model/backend work is
    /// completed before spawning children; subsequent rejection shuts them down.
    pub fn prepare_hybrid_cpu_tensor(
        resources: &BoundDecoderResources,
        segment: LayerSegmentPlan,
        parallel: HybridParallelPlan,
        lane: usize,
        config: PipelineConfig,
        options: &HybridCpuTensorConfig,
        endpoint: DecoderTensorCollective,
    ) -> Result<Self> {
        options.validate()?;
        config.validate()?;
        parallel.validate_spec(resources.spec())?;
        if parallel
            .expert_members()
            .iter()
            .any(|rank| parallel.tensor_members().contains(rank))
        {
            return Err(error("hybrid CPU expert children must not alias TP owners"));
        }
        admission::admit_owner(resources, &segment, &parallel, lane, config, options)?;
        let owner = parallel.tensor_members()[lane];
        let group = parallel
            .expert_attachment()
            .filter(|ep| ep.source_rank == owner)
            .map(|ep| ExpertGroup {
                source_rank: ep.source_rank,
                members: ep.members.clone(),
                layers: parallel.schema().layer_range(),
                placement: ep.placement.clone(),
                limits: options.expert_limits,
            });
        // Metadata-only support and peak retained storage checks precede any
        // module reads or EP worker startup, including on non-source TP lanes.
        reject_cpu_numeric_fp8(resources, parallel.schema().layer_range())?;
        if let Some(group) = &group {
            group.validate_resources(resources)?;
            for &member in &group.members {
                admission::admit_expert(resources, group, member, options.max_expert_weight_bytes)?;
            }
        }
        let layout = parallel.rank_layout(lane)?;
        let description = description(
            resources.spec(),
            segment.clone(),
            layout.state_schema(),
            config,
            group.clone(),
        )?;
        let capabilities = description.execution_capabilities()?;
        let planes = layout.kv_planes(config.page_size, config.max_positions)?;
        let control = endpoint.control();
        control.register_sites(owner, translate_hybrid_schedule(&parallel, &segment)?)?;
        let module = HybridTensorCpuSegment::prepare(
            resources,
            segment,
            parallel,
            lane,
            Box::new(endpoint),
            config.max_positions,
            config.max_parameter_bytes,
        )?;
        let backend = PagedKvBackend::new(TypedCpuPagedKvPool::from_strategy(
            &planes,
            config.physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))?,
        )?);
        let experts = if let Some(group) = group.clone() {
            let image = resources.clone();
            #[cfg(test)]
            let startup_hook = tests::EXPERT_STARTUP.with_borrow(Clone::clone);
            let mut executor = ExpertParallelExecutor::new(group, move |rank, group| {
                #[cfg(test)]
                if let Some(hook) = &startup_hook {
                    hook(rank)?;
                }
                ExpertRankWorker::prepare_cpu(&image, rank, group, config.max_parameter_bytes)
            })?;
            if let Err(source) = executor.set_dispatch_timeout(options.expert_timeout) {
                return Err(ferrule_common::Error::with_cleanup(
                    "expert timeout setup",
                    source,
                    executor.shutdown(),
                ));
            }
            Some(executor)
        } else {
            None
        };
        Self::new(
            TensorScopedStateProgram::new(
                HybridCpuTensorPipelineStageProgram::new(module, experts, group),
                control,
                owner,
            ),
            backend,
            description,
            capabilities,
        )
    }
}

impl PipelineParallelExecutor {
    /// Explicit reference factory; does not enable a CUDA/process hybrid route.
    /// Load metadata/resources on each persistent owner, not prepared model state.
    pub fn new_hybrid_cpu_tensor<F>(
        topology: ValidatedParallelTopology,
        resources_spec: &DecoderModelSpec,
        layout: HybridExpertTopology,
        config: PipelineConfig,
        options: HybridCpuTensorConfig,
        load: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank) -> Result<BoundDecoderResources> + Clone + Send + 'static,
    {
        build(
            topology,
            resources_spec,
            layout,
            config,
            options,
            load,
            |_, stage| Ok(stage),
        )
    }
}

// The private decoration seam lets tests observe real programs/backends, without
// replacing construction, model execution, the worker journal or parent forward.
fn build<F, D, P, B>(
    topology: ValidatedParallelTopology,
    spec: &DecoderModelSpec,
    layout: HybridExpertTopology,
    config: PipelineConfig,
    options: HybridCpuTensorConfig,
    load: F,
    decorate: D,
) -> Result<PipelineParallelExecutor>
where
    F: FnOnce(PipelineRank) -> Result<BoundDecoderResources> + Clone + Send + 'static,
    D: FnOnce(PipelineRank, HybridCpuTensorStage) -> Result<PipelineStage<B, P>>
        + Clone
        + Send
        + 'static,
    B: DecoderKvBackend<SequenceState = P::SequenceState>
        + KvCommitParticipant<
            SequenceState = P::SequenceState,
            Transaction = <B as DecoderKvBackend>::Transaction,
        > + 'static,
    P: PipelineStateProgram<KvView = <B as DecoderKvBackend>::KvView>,
{
    options.validate()?;
    let layout = HybridExpertTopology::new(spec, layout.stages().to_vec())?;
    let tp = topology.plan().tensor_parallel;
    let plans = layout
        .stages()
        .iter()
        .enumerate()
        .map(|(stage, p)| {
            LayerSegmentPlan::new(
                spec.layers().len(),
                p.schema().layer_range(),
                stage == 0,
                stage + 1 == layout.stages().len(),
            )
            .map_err(stage::segment_error)
        })
        .collect::<Result<Vec<_>>>()?;
    validate_pipeline_mesh(&topology, &plans, config, true)?;
    if layout.stages().len() != topology.plan().pipeline_parallel {
        return Err(error("hybrid PP stage count differs from KV mesh"));
    }
    let mut scopes = topology
        .execution_scopes(0)
        .map_err(|e| error(format!("hybrid KV scope: {e:?}")))?;
    let mut endpoints = Vec::new();
    let mut controls = Vec::new();
    let mut expected = Vec::new();
    let mut kv_layers = Vec::new();
    for (stage, parallel) in layout.stages().iter().enumerate() {
        if parallel.scope().topology != topology.topology_id()
            || parallel.scope().replica != 0
            || parallel.tensor_members()
                != scopes
                    .tensor_collective_participants(stage as u32, 0)
                    .map_err(|e| error(format!("hybrid TP scope: {e:?}")))?
                    .iter()
                    .collect::<Vec<_>>()
        {
            return Err(error(
                "hybrid TP roster must match the authenticated contiguous KV mesh",
            ));
        }
        let group = layout.optional_expert_group(stage, options.expert_limits)?;
        if let Some(group) = &group {
            if topology.plan().expert_parallel != 1
                && group.members.len() != topology.plan().expert_parallel
            {
                return Err(error(
                    "declared EP degree differs from explicit hybrid attachment",
                ));
            }
            if group
                .members
                .iter()
                .any(|rank| rank.get() < topology.world_size())
            {
                return Err(error(
                    "hybrid CPU EP owners must be disjoint from the entire KV mesh",
                ));
            }
        }
        if let Some(group) = &group {
            let attachment = ferrule_common::ExpertDispatchMembers::new_with_scope(
                &topology,
                parallel.scope().replica,
                parallel.scope().stage,
                ExpertSourceScope::ExternalTensorSource,
                group.source_rank,
                group.members.clone(),
            )
            .map_err(|e| error(format!("hybrid expert source scope: {e:?}")))?;
            scopes
                .attach_expert_dispatch_members(attachment)
                .map_err(|e| error(format!("hybrid expert attachment: {e:?}")))?;
        }
        let peers = DecoderTensorCollective::new_group_from_members(
            parallel.tensor_members(),
            topology.topology_id(),
            ParallelGroupId::new(stage as u32 + 1),
            options.collective_limits,
            options.collective_timeout,
        )?;
        controls.push(peers[0].control());
        endpoints.extend(peers.into_iter().map(Some));
        for lane in 0..tp {
            let local = parallel.rank_layout(lane)?;
            expected.push(description(
                spec,
                plans[stage].clone(),
                local.state_schema(),
                config,
                group
                    .clone()
                    .filter(|group| group.source_rank == local.owner()),
            )?);
            kv_layers.push(
                local
                    .state_schema()
                    .entries()
                    .iter()
                    .filter(|entry| entry.local_kv_layer.is_some())
                    .count(),
            );
        }
    }
    // Global logical page accounting comes from the actual model schema. Never
    // multiply replicated local KV heads by TP or build a synthetic dense spec.
    let logical_schema = HybridStateSchema::for_segment(spec, 0..spec.layers().len())?
        .kv_planes(config.page_size, config.max_positions)?;
    let endpoints = Arc::new(Mutex::new(endpoints));
    let factory = Arc::new(Mutex::new((load, decorate)));
    let checkpoint_binding = Arc::new(Mutex::new(None));
    let admitted = Arc::new(layout.clone());
    let model = spec.clone();
    let segments = plans.clone();
    let expected_owners = expected.clone();
    let degree = topology.world_size() as usize;
    let pool = DataParallelExecutor::new(
        DataParallelConfig {
            replicas: degree,
            max_outstanding_per_replica: 1,
            session_capacity: degree,
        },
        move |local| {
            let rank = PipelineParallelExecutor::rank(local.get() as usize);
            let stage_index = local.get() as usize / tp;
            let lane = local.get() as usize % tp;
            let (load, decorate) = factory
                .lock()
                .map_err(|_| error("hybrid factory lock poisoned"))?
                .clone();
            let resources = load(rank)?;
            if resources.spec() != &model {
                return Err(error("hybrid owner checkpoint changed after admission"));
            }
            seal_binding(&checkpoint_binding, &resources)?;
            let endpoint = endpoints
                .lock()
                .map_err(|_| error("hybrid endpoint lock poisoned"))?[local.get() as usize]
                .take()
                .ok_or_else(|| error("hybrid endpoint assigned twice"))?;
            let stage = HybridCpuTensorStage::prepare_hybrid_cpu_tensor(
                &resources,
                segments[stage_index].clone(),
                admitted.stages()[stage_index].clone(),
                lane,
                config,
                &options,
                endpoint,
            )?;
            let mut stage = decorate(rank, stage)?;
            if stage.description != expected_owners[local.get() as usize]
                || stage.program.hybrid_schema()
                    != Some(
                        admitted.stages()[stage_index]
                            .rank_layout(lane)?
                            .state_schema(),
                    )
            {
                return Err(ferrule_common::Error::with_cleanup(
                    "hybrid CPU owner identity",
                    error("returned owner differs from admitted layout"),
                    stage.shutdown(),
                ));
            }
            let boot = PipelineStageBoot {
                rank,
                plan: segments[stage_index].clone(),
                config,
                physical_pages: config
                    .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))?,
                program_spec: vec![],
            };
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        },
    )
    .map_err(pool_build_error)?;
    let mut transport = ThreadTransport::new(pool);
    let validation = (|| {
        for (index, expected) in expected.iter().enumerate() {
            match transport.call(PipelineParallelExecutor::rank(index), Command::Describe)? {
                Reply::Description(actual) if actual == *expected => (),
                _ => {
                    return Err(error(
                        "hybrid owner description differs from admitted schema/attachment",
                    ));
                }
            }
        }
        Ok(())
    })();
    if let Err(source) = validation {
        return Err(ferrule_common::Error::with_cleanup(
            "hybrid CPU parent startup",
            source,
            transport.shutdown(),
        ));
    }
    let mut pipeline = PipelineParallelExecutor::from_validated_transport(
        topology,
        config,
        transport,
        expected,
        scopes,
        Box::new(logical_schema),
        kv_layers,
    );
    pipeline.tensor_controls = controls;
    Ok(pipeline)
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/hybrid_tensor/mod.rs"]
mod tests;
