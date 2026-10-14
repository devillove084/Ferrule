//! Explicit hybrid PP2 seam. No runner, forward copy, or page authority.
use super::*;
use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::ModelExecutionBackend;
use ferrule_model::decoder::{
    HybridDecoderSequenceState, HybridSequenceLifecycle, HybridStateSchema, TypedCpuPagedKvPool,
};
use ferrule_model::transformer::{
    DecoderModelSpec, FeedForward, HybridDecoderSegment, HybridSegmentMemoryEstimate,
};

/// Validated model-derived partition, never inferred from legacy process descriptions.
#[derive(Clone, Debug)]
pub struct HybridPipelineLayout {
    schemas: Vec<HybridStateSchema>,
    groups: Option<Vec<ExpertGroup>>,
}
impl HybridPipelineLayout {
    pub fn new(spec: &DecoderModelSpec, plans: &[LayerSegmentPlan]) -> Result<Self> {
        if plans.len() != 2
            || spec
                .layers()
                .iter()
                .any(|l| !matches!(l.feed_forward(), FeedForward::SwiGlu(_)))
        {
            return Err(error(
                "dense hybrid layout requires PP2; routed stages need explicit expert attachments",
            ));
        }
        let schemas = plans
            .iter()
            .map(|plan| {
                if plan.total_layers() != spec.layers().len() {
                    return Err(error("hybrid plan layer count mismatch"));
                }
                HybridStateSchema::for_segment(spec, plan.layers())
            })
            .collect::<Result<Vec<_>>>()?;
        HybridStateSchema::validate_partition(spec, &schemas)?;
        Ok(Self {
            schemas,
            groups: None,
        })
    }
    /// TP1 roots remain the two KV participants; children are stage-scoped.
    pub fn with_routed_experts(
        resources: &BoundDecoderResources,
        plans: &[LayerSegmentPlan],
        groups: Vec<ExpertGroup>,
    ) -> Result<Self> {
        if plans.len() != 2 || groups.len() != 2 {
            return Err(error(
                "routed hybrid requires exactly two stage attachments",
            ));
        }
        let mut owners = BTreeSet::new();
        let mut schemas = Vec::new();
        for (index, (plan, group)) in plans.iter().zip(&groups).enumerate() {
            group.validate_resources(resources)?;
            if plan.total_layers() != resources.spec().layers().len()
                || group.layers != plan.layers()
                || group.source_rank != ParallelRankId::new(index as u32)
                || !matches!(group.members.len(), 2 | 4 | 8)
                || group
                    .members
                    .iter()
                    .any(|rank| rank.get() < 2 || !owners.insert(*rank))
            {
                return Err(error("routed hybrid source/range/actual owner mismatch"));
            }
            schemas.push(HybridStateSchema::for_segment(
                resources.spec(),
                plan.layers(),
            )?);
        }
        HybridStateSchema::validate_partition(resources.spec(), &schemas)?;
        Ok(Self {
            schemas,
            groups: Some(groups),
        })
    }
    pub fn schemas(&self) -> &[HybridStateSchema] {
        &self.schemas
    }
    pub fn stage_kv_layers(&self, stage: usize) -> usize {
        self.schemas[stage]
            .entries()
            .iter()
            .filter(|e| e.local_kv_layer.is_some())
            .count()
    }
    pub fn kv_layers(&self) -> usize {
        (0..self.schemas.len())
            .map(|i| self.stage_kv_layers(i))
            .sum()
    }
    pub(super) fn validate(
        &self,
        topology: &ValidatedParallelTopology,
        plans: &[LayerSegmentPlan],
        config: PipelineConfig,
    ) -> Result<()> {
        if self.groups.is_some() {
            validate_pipeline_mesh(topology, plans, config, false)?;
        } else {
            validate_pipeline(topology, plans, config)?;
        }
        let p = topology.plan();
        if p.pipeline_parallel != 2
            || p.tensor_parallel != 1
            || self
                .groups
                .as_ref()
                .map_or(p.expert_parallel != 1, |groups| {
                    groups.iter().any(|g| g.members.len() != p.expert_parallel)
                })
            || config.precision != ExecutionPrecisionPolicy::f32()
            || plans.len() != self.schemas.len()
            || plans
                .iter()
                .zip(&self.schemas)
                .any(|(p, s)| p.layers() != s.layer_range() || p.total_layers() != s.total_layers())
        {
            return Err(error(
                "hybrid requires F32 PP2 TP1 with exact schemas and declared expert attachments",
            ));
        }
        Ok(())
    }
    /// Caller provides actual per-owner weight/workspace/margin estimates. Tied
    /// endpoint residency is charged per owner, never deduplicated across owners.
    pub fn stage_budget(
        &self,
        stage: usize,
        config: PipelineConfig,
        device: usize,
        weights: u64,
        workspace: u64,
        margin: u64,
        backend: ModelExecutionBackend,
    ) -> Result<HybridStageBudget> {
        config.validate()?;
        if config.precision != ExecutionPrecisionPolicy::f32() {
            return Err(error("hybrid budget requires F32"));
        }
        let schema = self
            .schemas
            .get(stage)
            .ok_or_else(|| error("invalid hybrid budget stage"))?;
        let (state, kv) = stage_capacity(
            config,
            schema.per_sequence_state_bytes()?,
            schema
                .kv_planes(config.page_size, config.max_positions)?
                .checked_page_bytes()
                .ok_or_else(|| error("hybrid KV budget overflow"))?,
            PhysicalKvLayout::hybrid(backend),
        )?;
        let budget = HybridStageBudget {
            device,
            weights,
            state,
            kv,
            workspace,
            margin,
        };
        budget.total()?;
        Ok(budget)
    }
}
/// Runtime owns multiplicity, not model geometry or parameter representation.
fn stage_capacity(
    config: PipelineConfig,
    state_unit: usize,
    kv_unit: usize,
    layout: PhysicalKvLayout,
) -> Result<(u64, u64)> {
    config.validate()?;
    if config.precision != ExecutionPrecisionPolicy::f32() {
        return Err(error("hybrid budget requires F32"));
    }
    let copies = config
        .session_capacity
        .checked_add(1)
        .ok_or_else(|| error("hybrid session capacity overflow"))?;
    let state = state_unit
        .checked_mul(copies)
        .ok_or_else(|| error("hybrid state budget overflow"))?;
    // Both hybrid adapters use shadow/COW custody: physical pages = 2L.
    let kv = kv_unit
        .checked_mul(config.physical_pages(layout)?)
        .ok_or_else(|| error("hybrid KV budget overflow"))?;
    Ok((state as u64, kv as u64))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridStageBudget {
    pub device: usize,
    pub weights: u64,
    pub state: u64,
    pub kv: u64,
    pub workspace: u64,
    pub margin: u64,
}
impl HybridStageBudget {
    /// Compose model byte units once with runtime capacity and placement policy.
    /// CUDA host bindings are additional and must also enter host admission.
    pub fn from_segment_estimate(
        config: PipelineConfig,
        device: usize,
        estimate: &HybridSegmentMemoryEstimate,
        backend: ModelExecutionBackend,
        margin: u64,
    ) -> Result<Self> {
        estimate.validate_execution_limits(
            config.page_size,
            config.max_positions,
            config.max_batch_tokens,
        )?;
        let memory = match backend {
            ModelExecutionBackend::Cpu => estimate.host,
            ModelExecutionBackend::Cuda => estimate.device,
        };
        let (state, kv) = stage_capacity(
            config,
            estimate.per_sequence_state_bytes,
            estimate.per_page_kv_bytes,
            PhysicalKvLayout::hybrid(backend),
        )?;
        let budget = Self {
            device,
            weights: memory.weight_bytes()? as u64,
            workspace: memory.workspace_bytes as u64,
            state,
            kv,
            margin,
        };
        budget.total()?;
        Ok(budget)
    }
    pub fn total(&self) -> Result<u64> {
        [
            self.weights,
            self.state,
            self.kv,
            self.workspace,
            self.margin,
        ]
        .into_iter()
        .try_fold(0u64, |a, b| {
            a.checked_add(b)
                .ok_or_else(|| error("hybrid device budget overflow"))
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridDeviceBudget {
    pub device: usize,
    pub required: u64,
    pub capacity: u64,
}
impl HybridDeviceBudget {
    /// One authoritative capacity sample per physical device. Check before owner allocation.
    pub fn admit(stages: &[HybridStageBudget], capacities: &[(usize, u64)]) -> Result<Vec<Self>> {
        let mut samples = BTreeMap::new();
        for &(device, capacity) in capacities {
            if samples.insert(device, capacity).is_some() {
                return Err(error("duplicate hybrid device capacity"));
            }
        }
        let mut totals = BTreeMap::<usize, u64>::new();
        for stage in stages {
            let total = totals.entry(stage.device).or_default();
            *total = total
                .checked_add(stage.total()?)
                .ok_or_else(|| error("colocated hybrid budget overflow"))?;
        }
        if totals.len() != samples.len() || totals.is_empty() {
            return Err(error("hybrid capacity samples must exactly cover devices"));
        }
        totals
            .into_iter()
            .map(|(device, required)| {
                let capacity = *samples
                    .get(&device)
                    .ok_or_else(|| error("missing hybrid device capacity"))?;
                if required > capacity {
                    return Err(error(
                        "colocated hybrid owners exceed physical device capacity",
                    ));
                }
                Ok(Self {
                    device,
                    required,
                    capacity,
                })
            })
            .collect()
    }
}

pub struct HybridCpuPipelineStageProgram {
    segment: HybridDecoderSegment,
}
impl PipelineStateProgram for HybridCpuPipelineStageProgram {
    type SequenceState = HybridDecoderSequenceState;
    type Lifecycle = HybridSequenceLifecycle;
    type KvView = ferrule_model::decoder::CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        Some(self.segment.schema())
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        Ok(HybridSequenceLifecycle::new(self.segment.schema().clone()))
    }
    fn execute_state(
        &mut self,
        batch: &ferrule_model::decoder::PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        if context.experts.is_some() {
            return Err(error("hybrid PPxEP unsupported"));
        }
        (context.check_active)(context.transaction)?;
        self.segment
            .execute_checked(batch, input, states, view, context.check_active)
            .map_err(stage::segment_error)
    }
    fn validate_frontier(
        &self,
        batch: &ferrule_model::decoder::PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.segment
            .validate_frontier(batch, states)
            .map_err(stage::segment_error)
    }
}
impl
    PipelineStage<
        PagedKvBackend<TypedCpuPagedKvPool<HybridDecoderSequenceState>>,
        HybridCpuPipelineStageProgram,
    >
{
    pub fn prepare_hybrid_cpu(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.precision != ExecutionPrecisionPolicy::f32()
            || resources
                .spec()
                .layers()
                .iter()
                .any(|l| !matches!(l.feed_forward(), FeedForward::SwiGlu(_)))
        {
            return Err(error("hybrid CPU pipeline requires dense F32"));
        }
        let schema = HybridStateSchema::for_segment(resources.spec(), plan.layers())?;
        let planes = schema.kv_planes(config.page_size, config.max_positions)?;
        let (kv_heads, head_dim) = schema
            .layers()
            .iter()
            .find_map(|l| match l {
                ferrule_model::decoder::HybridLayerSchema::FullAttention {
                    kv_heads,
                    head_dim,
                    ..
                } => Some((*kv_heads, *head_dim)),
                _ => None,
            })
            .ok_or_else(|| error("hybrid stage has no full attention"))?;
        let physical_pages = config.physical_pages(PhysicalKvLayout::hybrid(
            ferrule_model::ModelExecutionBackend::Cpu,
        ))?;
        let description = Description {
            plan: plan.clone(),
            physical_pages,
            config,
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            kv_heads,
            head_dim,
            expert_group: None,
        };
        let capabilities = description.execution_capabilities()?;
        let segment = HybridDecoderSegment::prepare(
            resources,
            plan,
            config.precision,
            config.max_positions,
            config.max_parameter_bytes,
        )
        .map_err(stage::segment_error)?;
        let backend =
            PagedKvBackend::new(TypedCpuPagedKvPool::from_strategy(&planes, physical_pages)?);
        Self::new(
            HybridCpuPipelineStageProgram { segment },
            backend,
            description,
            capabilities,
        )
    }
}
impl PipelineParallelExecutor {
    /// Explicit thread-only hybrid factory. Model objects are constructed inside
    /// owners; the parent retains just validated schemas and the sole page manager.
    pub fn new_hybrid_with_program<F, B, P>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        layout: HybridPipelineLayout,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank, LayerSegmentPlan) -> Result<PipelineStage<B, P>>
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
        layout.validate(&topology, &plans, config)?;
        let owner_plans = plans.clone();
        let schemas = layout.schemas.clone();
        let groups = layout.groups.clone();
        let external = groups.is_some();
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 2,
                max_outstanding_per_replica: 1,
                session_capacity: 2,
            },
            move |local| {
                let index = local.get() as usize;
                let rank = Self::rank(index);
                let plan = owner_plans[index].clone();
                let mut stage = factory(rank, plan.clone())?;
                if stage.program.hybrid_schema() != Some(&schemas[index])
                    || stage.description.expert_group.as_ref() != groups.as_ref().map(|g| &g[index])
                {
                    return Err(ferrule_common::Error::with_cleanup(
                        "hybrid owner startup",
                        error("owner hybrid schema/EP mismatch"),
                        stage.shutdown(),
                    ));
                }
                let boot = PipelineStageBoot {
                    rank,
                    plan,
                    physical_pages: config.physical_pages(PhysicalKvLayout::hybrid(
                        ferrule_model::ModelExecutionBackend::Cpu,
                    ))?,
                    config,
                    program_spec: Vec::new(),
                };
                Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
            },
        )
        .map_err(pool_build_error)?;
        Self::new_with_transport_inner(
            topology,
            plans,
            config,
            ThreadTransport::new(pool),
            false,
            if external {
                ExpertSourceScope::ExternalStage
            } else {
                ExpertSourceScope::Member
            },
            Some(layout),
        )
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/hybrid_budget/mod.rs"]
mod budget_tests;

#[cfg(unix)]
impl PipelineParallelExecutor {
    /// Explicit process hybrid composition. Metadata is rebound and checked by each
    /// child; only this executor owns the logical page manager and publication.
    /// As with the thread constructor, callers must admit aggregate host/device
    /// residency first. Child CUDA budgets are checked again against live capacity.
    pub fn new_hybrid_process(
        topology: ValidatedParallelTopology,
        config: PipelineConfig,
        resources: &BoundDecoderResources,
        launch: crate::parallel::process::ProcessLaunch,
        boots: Vec<(
            crate::parallel::process::ProcessIdentity,
            crate::parallel::process::decoder::DecoderBoot,
        )>,
        options: crate::parallel::process::ProcessOwnerConfig,
    ) -> Result<Self> {
        use crate::parallel::process::decoder::ProcessPipelineTransport;
        let plans = boots
            .iter()
            .map(|(_, boot)| boot.segment.decode())
            .collect::<Result<Vec<_>>>()?;
        let layout = HybridPipelineLayout::new(resources.spec(), &plans)?;
        layout.validate(&topology, &plans, config)?;
        options.validate().map_err(error)?;
        for (index, (identity, boot)) in boots.iter().enumerate() {
            let stage = boot.stage_boot()?;
            if identity.rank.get() as usize != index
                || boot.rank as usize != index
                || stage.config != config
            {
                return Err(error("hybrid process owner order/config mismatch"));
            }
            boot.hybrid
                .as_ref()
                .ok_or_else(|| error("hybrid process requires explicit schema"))?
                .validate_resources(boot, resources)?;
            boot.validate_frame_limits(options.frame_limits)?;
        }
        let transport = ProcessPipelineTransport::spawn_hybrid_admitted(launch, boots, options)?;
        Self::new_with_transport_inner(
            topology,
            plans,
            config,
            transport,
            false,
            ExpertSourceScope::Member,
            Some(layout),
        )
    }
}
