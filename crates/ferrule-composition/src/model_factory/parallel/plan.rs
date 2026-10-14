//! Payload-free owner geometry. Family capability and provider admission are separate.
use super::experts::OwnerPlan;
use super::*;
use ferrule_model::transformer::DecoderModelSpec;
use ferrule_model::transformer::hybrid_parallel::{
    HybridRankPrecision, HybridRankStorageBudget, HybridRoutedExpertExecution,
};

/// Ownership is explicit, never inferred from a device count or numeric profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelExpertMode {
    LocalTensorShards,
    ExternalAttachments,
}

/// Immutable provider intent. Construction describes capabilities, not permission
/// to execute; the model validator and complete provider admission remain required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelExecutionContract {
    precision: HybridRankPrecision,
    experts: ParallelExpertMode,
}
impl ParallelExecutionContract {
    pub const fn new(precision: HybridRankPrecision, experts: ParallelExpertMode) -> Self {
        Self { precision, experts }
    }
    pub const fn external_f32() -> Self {
        Self::new(
            HybridRankPrecision::F32,
            ParallelExpertMode::ExternalAttachments,
        )
    }
    pub const fn precision(self) -> HybridRankPrecision {
        self.precision
    }
    pub const fn experts(self) -> ParallelExpertMode {
        self.experts
    }
    fn validate(self, options: &PipelineBuildOptions) -> Result<()> {
        use crate::model_factory::execution_policy::*;
        let p = options.parallelism;
        match self.experts {
            ParallelExpertMode::LocalTensorShards
                if p.expert_parallel != 1 || p.tensor_parallel != 2 =>
            {
                return Err(invalid("local expert shards require TP2 EP1"));
            }
            ParallelExpertMode::ExternalAttachments if p.expert_parallel <= 1 => {
                return Err(invalid("external expert ownership requires EP > 1"));
            }
            _ => {}
        }
        if self.precision.is_mixed() && p.tensor_parallel != 2 {
            return Err(invalid(
                "Mixed metadata requires TP2; no TP4 precision inference",
            ));
        }
        options.validate_policy(
            ferrule_model::ModelExecutionBackend::Cuda,
            match self.experts {
                ParallelExpertMode::LocalTensorShards => OwnerLayout::StageRoots,
                ParallelExpertMode::ExternalAttachments => OwnerLayout::StageRootsWithExperts,
            },
            ExecutionCapabilities {
                ranks: RankCapability::ThreadOnly,
                tensor: TensorCapability::DeviceThreads { degrees: &[2, 4] },
            },
            PlacementPolicy {
                selection: DeviceSelection::Explicit,
                sharing: DeviceSharing::Colocated,
            },
        )?;
        Ok(())
    }
    fn validate_execution(self, layout: &HybridExpertTopology) -> Result<()> {
        for stage in layout.stages() {
            stage.validate_rank_execution()?;
        }
        // Storage-only metadata must never be passed off as full provider admission
        // when model execution is enabled independently by another integration.
        if self.precision.is_mixed() || self.experts == ParallelExpertMode::LocalTensorShards {
            return Err(invalid(
                "parallel typed execution requires complete model provider/workspace requirements and validated runtime precision schedule",
            ));
        }
        Ok(())
    }
}

/// Sealed, payload-free lowering reused by provider assembly. No worker, CUDA
/// context, checkpoint payload or host expert cache is constructed here.
pub struct ParallelExecutionPlan {
    geometry: Geometry,
    layout: HybridExpertTopology,
    contract: ParallelExecutionContract,
}
impl ParallelExecutionPlan {
    pub fn describe(
        spec: &DecoderModelSpec,
        options: &PipelineBuildOptions,
        contract: ParallelExecutionContract,
        attachments: Vec<HybridExpertAttachment>,
    ) -> Result<Self> {
        contract.validate(options)?;
        let geometry = Geometry::new(spec.layers().len(), options)?;
        Self::assemble(spec, geometry, contract, attachments)
    }
    fn assemble(
        spec: &DecoderModelSpec,
        geometry: Geometry,
        contract: ParallelExecutionContract,
        attachments: Vec<HybridExpertAttachment>,
    ) -> Result<Self> {
        let external = contract.experts == ParallelExpertMode::ExternalAttachments;
        if attachments.len() != if external { geometry.segments.len() } else { 0 } {
            return Err(invalid(
                "expert attachments differ from sealed ownership mode",
            ));
        }
        let stages = geometry
            .segments
            .iter()
            .enumerate()
            .map(|(stage, segment)| {
                let routed = if external {
                    HybridRoutedExpertExecution::ExternalExpertAttachment(
                        attachments[stage].clone(),
                    )
                } else {
                    HybridRoutedExpertExecution::LocalTensorShards
                };
                Ok(HybridParallelPlan::new_with_execution(
                    spec,
                    segment.layers(),
                    HybridExpertScope {
                        topology: geometry.topology.topology_id(),
                        replica: 0,
                        stage: stage as u32,
                    },
                    geometry.root_members(stage)?,
                    routed,
                    contract.precision,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let layout = HybridExpertTopology::new(spec, stages)?;
        if external {
            layout
                .validate_external_expert_roster(geometry.parallelism)
                .map_err(|e| invalid(e.to_string()))?;
        }
        Ok(Self {
            geometry,
            layout,
            contract,
        })
    }
    /// Execution preflight, not metadata support or a live GPU reservation.
    pub fn prepare(
        spec: &DecoderModelSpec,
        options: &PipelineBuildOptions,
        contract: ParallelExecutionContract,
        attachments: Vec<HybridExpertAttachment>,
    ) -> Result<Self> {
        let plan = Self::describe(spec, options, contract, attachments)?;
        plan.validate_execution()?;
        Ok(plan)
    }
    pub fn validate_execution(&self) -> Result<()> {
        self.contract.validate_execution(&self.layout)
    }
    pub fn contract(&self) -> ParallelExecutionContract {
        self.contract
    }
    pub fn topology(&self) -> &ferrule_common::ValidatedParallelTopology {
        &self.geometry.topology
    }
    pub fn layout(&self) -> &HybridExpertTopology {
        &self.layout
    }
    pub fn devices(&self) -> &[usize] {
        &self.geometry.devices
    }
    pub fn segments(&self) -> &[LayerSegmentPlan] {
        &self.geometry.segments
    }
    pub fn root_count(&self) -> usize {
        self.geometry.roots
    }
    pub fn external_owner_count(&self) -> usize {
        self.geometry.devices.len() - self.geometry.roots
    }

    /// Bind an explicit BF16 block schedule to this stage's roots. These host
    /// payload claims neither create EP owners nor complete device admission.
    pub fn collective_claims(
        &self,
        stage: usize,
        collective: &ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan,
    ) -> Result<ParallelCollectiveClaims> {
        let rank_plan = self
            .layout
            .stages()
            .get(stage)
            .ok_or_else(|| invalid("unknown collective stage"))?;
        if !self.contract.precision.is_mixed()
            || collective.epoch() != rank_plan.scope().topology
            || collective.members() != rank_plan.tensor_members()
            || !rank_plan.tensor_members().contains(&collective.source())
            || collective.sites().iter().any(|site| {
                site.layer
                    .is_some_and(|layer| !rank_plan.schema().layer_range().contains(&layer))
            })
        {
            return Err(invalid(
                "typed collective differs from sealed stage precision/roster/source/layers",
            ));
        }
        let payload = collective.payload_budget();
        let route_control = collective.route_control_budget()?;
        let owners = collective
            .members()
            .iter()
            .map(|&owner| ParallelRankCollectiveClaim {
                owner,
                device: self.geometry.devices[owner.get() as usize],
                host_input_bytes: payload.input_bytes_per_owner,
                host_output_bytes: payload.output_bytes_per_owner,
                host_route_record_bytes: route_control.payload_bytes_per_owner,
            })
            .collect();
        Ok(ParallelCollectiveClaims {
            stage,
            source: collective.source(),
            service_host_bytes: payload.service_peak_bytes,
            route_control_host_bytes: route_control.total_host_bytes,
            total_host_bytes: payload.total_host_bytes,
            owners,
        })
    }

    /// Storage components only; intentionally cannot produce HybridStageBudget or
    /// a complete device total until the model exports the missing provider ledger.
    pub fn storage_claims(
        &self,
        resources: &BoundDecoderResources,
        page_size: usize,
        max_positions: usize,
        physical_pages: usize,
        sessions: usize,
    ) -> Result<Vec<ParallelRankStorageClaim>> {
        let copies = sessions
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("state copies overflow"))?;
        if sessions == 0 {
            return Err(invalid("empty parallel session capacity"));
        }
        let mut claims = Vec::new();
        for (stage, segment) in self.layout.stages().iter().zip(&self.geometry.segments) {
            for lane in 0..stage.tensor_members().len() {
                let rank = stage.rank_layout(lane)?;
                claims.push(ParallelRankStorageClaim {
                    owner: rank.owner(),
                    device: self.geometry.devices[rank.owner().get() as usize],
                    storage: rank.storage_budget(
                        resources,
                        segment,
                        page_size,
                        max_positions,
                        physical_pages,
                        copies,
                    )?,
                });
            }
        }
        Ok(claims)
    }
}
#[derive(Debug)]
pub struct ParallelRankStorageClaim {
    pub owner: ParallelRankId,
    pub device: usize,
    pub storage: HybridRankStorageBudget,
}

/// Host payload accounting only. Device transport/pinned buffers, operator
/// providers, state/KV and allocator margins are not inferred from these values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelRankCollectiveClaim {
    pub owner: ParallelRankId,
    pub device: usize,
    pub host_input_bytes: usize,
    pub host_output_bytes: usize,
    pub host_route_record_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelCollectiveClaims {
    pub stage: usize,
    pub source: ParallelRankId,
    pub service_host_bytes: usize,
    /// Included in total_host_bytes; must not be charged a second time.
    pub route_control_host_bytes: usize,
    pub total_host_bytes: usize,
    pub owners: Vec<ParallelRankCollectiveClaim>,
}

pub(in crate::model_factory) struct Geometry {
    pub(in crate::model_factory) segments: Vec<LayerSegmentPlan>,
    pub(in crate::model_factory) devices: Vec<usize>,
    pub(in crate::model_factory) roots: usize,
    parallelism: ferrule_common::ParallelismPlan,
    pub(in crate::model_factory) topology: ferrule_common::ValidatedParallelTopology,
}
impl Geometry {
    pub(in crate::model_factory) fn new(
        layers: usize,
        options: &PipelineBuildOptions,
    ) -> Result<Self> {
        let p = options.parallelism;
        p.validate().map_err(|e| invalid(e.to_string()))?;
        if layers == 0 || p.pipeline_parallel > layers {
            return Err(invalid("pipeline stages require nonempty layer ranges"));
        }
        let count = options.owner_count()?;
        let roots = p
            .pipeline_parallel
            .checked_mul(p.tensor_parallel)
            .filter(|n| *n > 0 && u32::try_from(*n).is_ok())
            .ok_or_else(|| invalid("root roster exceeds rank ABI"))?;
        let devices = options
            .devices
            .clone()
            .unwrap_or_else(|| (0..count).collect());
        if devices.len() != count || devices.iter().any(|&d| i32::try_from(d).is_err()) {
            return Err(invalid("device placement differs from owner roster"));
        }
        let topology = ferrule_common::ValidatedParallelTopology::new(
            ferrule_common::ParallelTopologyId::new(1),
            roots as u32,
            ParallelRankId::new(0),
            p,
        )
        .map_err(|e| invalid(format!("parallel topology: {e:?}")))?;
        Ok(Self {
            topology,
            segments: super::super::pipeline::segment_plans(layers, p.pipeline_parallel)?,
            devices,
            roots,
            parallelism: p,
        })
    }

    pub(in crate::model_factory) fn root_members(
        &self,
        stage: usize,
    ) -> Result<Vec<ParallelRankId>> {
        if stage >= self.segments.len() {
            return Err(invalid("unknown pipeline stage"));
        }
        let start = stage * self.parallelism.tensor_parallel;
        Ok((start..start + self.parallelism.tensor_parallel)
            .map(|r| ParallelRankId::new(r as u32))
            .collect())
    }
    pub(in crate::model_factory) fn expert_members(
        &self,
        stage: usize,
    ) -> Result<Vec<ParallelRankId>> {
        if stage >= self.segments.len() {
            return Err(invalid("unknown expert stage"));
        }
        let degree = self.parallelism.expert_parallel;
        if degree == 1 {
            return Ok(Vec::new());
        }
        let start = self.roots + stage * degree;
        Ok((start..start + degree)
            .map(|r| ParallelRankId::new(r as u32))
            .collect())
    }
}
pub(in crate::model_factory) struct Plan {
    geometry: Geometry,
    pub(in crate::model_factory) segment_layout: Option<HybridPipelineLayout>,
    pub(in crate::model_factory) layout: HybridExpertTopology,
    config: PipelineConfig,
    pub(in crate::model_factory) kv: ferrule_runtime::engine::ResidentKvPagePlan,
    pub(in crate::model_factory) owners: Vec<OwnerPlan>,
    provider: NumericProviderInputs,
    execution: ParallelExecutionContract,
    pub(in crate::model_factory) wire_bytes: u64,
    pub(in crate::model_factory) collective: HostCollectiveLimits,
}
impl Plan {
    pub(in crate::model_factory) fn validate_execution(&self) -> Result<()> {
        self.execution.validate_execution(&self.layout)
    }
    pub(super) fn external_owner(&self, stage: usize) -> Option<&OwnerPlan> {
        self.owners.iter().find(|o| {
            self.layout
                .stages()
                .get(stage)
                .is_some_and(|s| s.schema().layer_range() == o.group.layers)
        })
    }
    pub(in crate::model_factory) fn config(&self) -> PipelineConfig {
        self.config
    }

    pub(in crate::model_factory) fn profile(&self) -> NumericFp8Profile {
        self.provider.numeric_profile
    }

    pub(in crate::model_factory) fn host(&self) -> HostExpertCacheOptions {
        self.provider.host_cache
    }

    /// Root options and expert startup consume the same sealed provider facts.
    /// No request or replacement inputs can be supplied at the admission boundary.
    pub(in crate::model_factory) fn decoder_options(
        &self,
        resources: &BoundDecoderResources,
    ) -> Result<ferrule_model::decoder::GenericDecoderOptions> {
        self.validate_execution()?;
        super::budget::decoder_options(
            resources,
            self.provider.family.clone(),
            self.provider.limits,
            self.config.max_parameter_bytes,
            self.profile(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(super) fn charge_pinned_host(&mut self, bytes: u64) -> Result<()> {
        // TP admission augments the ledger without reopening host policy or limits.
        let budget = &mut self.provider.host_cache.memory_budget;
        budget.base_resident_bytes = add(budget.base_resident_bytes, bytes)?;
        Ok(())
    }

    pub(in crate::model_factory) fn new(
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
        inputs: &NumericProviderInputs,
    ) -> Result<Self> {
        if request.backend != ferrule_model::ModelExecutionBackend::Cuda {
            return Err(invalid("numeric parallel provider requires CUDA"));
        }
        if inputs.limits.max_positions != request.driver_config.ctx_size
            || inputs.limits.max_batch_tokens != request.scheduler_config.max_batch_tokens
            || inputs.limits.max_sequences != request.scheduler_config.max_active_sequences
        {
            return Err(invalid(
                "numeric provider limits differ from pipeline context, batch or sequence capacities",
            ));
        }
        super::bindings::validate_bound(request, resources)?;
        crate::model_factory::parallel::bindings::validate_options(request)?;
        let parallel = request
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid("missing numeric pipeline placement"))?;
        if inputs.placement.is_some()
            || inputs.host_cache.mode
                != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
        {
            return Err(invalid(
                "pipeline numeric provider requires full shared prewarm and stage placement, not a resident placement",
            ));
        }
        inputs.parallel_execution.validate(parallel)?;
        let p = parallel.parallelism;
        let spec = resources.spec();
        let geometry = Geometry::new(request.max_layers, parallel)?;
        let segments = &geometry.segments;
        let owners = segments
            .iter()
            .enumerate()
            .filter(|_| {
                inputs.parallel_execution.experts() == ParallelExpertMode::ExternalAttachments
            })
            .map(|(stage, segment)| {
                OwnerPlan::for_stage_degree(
                    resources,
                    p.expert_parallel,
                    request.scheduler_config.max_batch_tokens,
                    segment.layers(),
                    geometry.root_members(stage)?[0],
                    geometry.expert_members(stage)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let attachments = owners
            .iter()
            .map(|owner| HybridExpertAttachment {
                source_rank: owner.group.source_rank,
                members: owner.group.members.clone(),
                placement: owner.group.placement.clone(),
            })
            .collect();
        let metadata = ParallelExecutionPlan::assemble(
            spec,
            geometry,
            inputs.parallel_execution,
            attachments,
        )?;
        let ParallelExecutionPlan {
            geometry,
            layout,
            contract: execution,
        } = metadata;
        // Never feed storage-only Mixed/local metadata into the legacy full F32
        // estimator. The public description remains available without this gate.
        execution.validate_execution(&layout)?;
        let segments = &geometry.segments;
        let segment_layout = if p.tensor_parallel == 1 {
            Some(HybridPipelineLayout::with_routed_experts(
                resources,
                segments,
                owners.iter().map(|o| o.group.clone()).collect(),
            )?)
        } else {
            None
        };
        let mut physical_page_bytes = 0;
        for stage in layout.stages() {
            for lane in 0..p.tensor_parallel {
                let planes = stage
                    .rank_layout(lane)?
                    .kv_planes(16, request.driver_config.ctx_size)?;
                physical_page_bytes =
                    add(physical_page_bytes, super::physical_page_bytes(&planes)?)?;
            }
        }
        let planes = HybridStateSchema::from_spec(spec, request.max_layers)?
            .kv_planes(16, request.driver_config.ctx_size)?;
        let kv = ferrule_runtime::engine::plan_pipeline_kv_pages(
            &planes,
            physical_page_bytes,
            request.kv_cache_bytes,
            PhysicalKvLayout::hybrid(request.backend),
            request.scheduler_config,
            request.driver_config,
        )?;
        let config = PipelineConfig {
            page_size: 16,
            max_pages: kv.full_capacity_pages,
            max_positions: request.driver_config.ctx_size,
            max_batch_tokens: request.scheduler_config.max_batch_tokens,
            session_capacity: request.scheduler_config.max_active_sequences,
            max_parameter_bytes: request.max_tensor_bytes,
            precision: if execution.precision.is_mixed() {
                ExecutionPrecisionPolicy::bf16_compatibility()
            } else {
                ExecutionPrecisionPolicy::f32()
            },
            max_ack_polls: 8,
        };
        config.validate()?;
        let collective = super::budget::collective_limits(&layout, segments, config)?;
        let wire_bytes = super::budget::wire_bytes(
            config.max_batch_tokens,
            spec.hidden_size(),
            spec.vocab_size(),
            &owners,
            if p.tensor_parallel == 1 {
                1
            } else {
                geometry.roots
            },
        )?;
        let host = super::budget::host_options(
            resources,
            inputs.host_cache,
            &geometry,
            config,
            kv,
            wire_bytes,
            collective,
            p.tensor_parallel > 1,
        )?;
        Ok(Self {
            geometry,
            execution,
            segment_layout,
            layout,
            config,
            kv,
            owners,
            provider: NumericProviderInputs {
                host_cache: host,
                ..inputs.clone()
            },
            wire_bytes,
            collective,
        })
    }
}

impl std::ops::Deref for Plan {
    type Target = Geometry;
    fn deref(&self) -> &Geometry {
        &self.geometry
    }
}
