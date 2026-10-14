//! Catalog-owned lowering. Construction, option adaptation and metadata admission
//! travel together; shared planning never rediscovers a provider from arithmetic.
use super::*;

type EngineBuilder = fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine>;

#[derive(Debug, Clone, Copy)]
pub(super) enum ResolvedConstruction {
    Resident(EngineBuilder),
    Pipeline(EngineBuilder),
}
impl ResolvedConstruction {
    pub(super) const fn kind(self) -> SelectedImplementationKind {
        match self {
            Self::Resident(_) => SelectedImplementationKind::Resident,
            Self::Pipeline(_) => SelectedImplementationKind::Pipeline,
        }
    }
    pub(super) const fn builder(self) -> EngineBuilder {
        match self {
            Self::Resident(build) | Self::Pipeline(build) => build,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PipelineConversion {
    Allowed,
    Frozen(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NumericModelRequest {
    pub(super) storage: NumericStoragePolicy,
    pub(super) precision: ferrule_model::transformer::NumericFp8Precision,
}

/// Complete normalized request passed to every catalog selector. Normalization
/// does not choose a constructor, read a checkpoint, or erase explicit options.
#[derive(Debug, Clone, Default)]
pub(super) struct SelectionRequest {
    pub(super) serving: ServingModelOptions,
    pub(super) numeric_model: Option<NumericModelRequest>,
}
impl SelectionRequest {
    pub(super) fn parallel(parallel: Option<PipelineBuildOptions>) -> Self {
        Self {
            serving: ServingModelOptions {
                parallel,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    pub(super) fn normalize(
        mut self,
        backend: BackendSelection,
        options: &ModelFactoryOptions,
    ) -> Result<Self> {
        expert_read_policy::validate_request(
            backend,
            options,
            &self.serving,
            self.numeric_model.is_some(),
        )?;
        if self.numeric_model.is_some() && self.serving.model_profile.is_some() {
            return Err(invalid(
                "named profile already selects numeric model precision; omit the compatibility override",
            ));
        }
        if let Some(numeric) = self.numeric_model {
            bf16_serving::validate_numeric_request(
                backend,
                options,
                &self.serving,
                numeric.storage,
                numeric.precision,
            )?;
        } else {
            self.serving.validate_before_model_io(
                backend,
                options,
                &ServingOptionProvenance::default(),
            )?;
        }
        if let Some(profile) = self.serving.model_profile {
            self.numeric_model = Some(profile.lower(&mut self.serving)?);
        }
        if self.serving.parallel.is_none()
            && self.serving.engine == ServingEngineSelection::Pipeline
        {
            self.serving.parallel = Some(PipelineBuildOptions::default());
        }
        Ok(self)
    }
    pub(super) fn reject_specialized(&self) -> Result<()> {
        if self.numeric_model.is_some() || self.serving.model_profile.is_some() {
            return Err(invalid(
                "selected model does not implement the requested numeric model profile",
            ));
        }
        if self.serving.model_precision == Some(ModelPrecision::Bf16) {
            return Err(invalid(
                "requires native BF16 dense Qwen3 Safetensors; no GDN/MoE/Qwen3.5",
            ));
        }
        if self.serving.kv_execution_policy != ferrule_runtime::engine::KvExecutionPolicy::Resident
        {
            return Err(invalid(
                "HostKV decode serving requires explicit CUDA, full-depth dense Qwen3; no GDN/numeric/TP/EP",
            ));
        }
        Ok(())
    }
}

pub(super) type StrategySelector = fn(
    &ModelImplementation,
    BackendSelection,
    &SelectionRequest,
    &ModelFactoryOptions,
) -> Result<ResolvedStrategy>;

#[derive(Debug)]
pub(super) struct ModelOptionPolicy {
    pub(super) serving: fn(
        &ModelFamily,
        &ServingModelOptions,
        ModelFactoryOptions,
        ResolvedConstruction,
    ) -> Result<ModelFactoryOptions>,
    pub(super) parallel: fn(
        ResidentModelBuildPlan,
        PipelineBuildOptions,
        ResolvedExecution,
    ) -> Result<ResidentModelBuildPlan>,
    pub(super) adapt: fn(
        &ModelDescriptor,
        &mut ModelFactoryOptions,
        ResolvedExecution,
    ) -> Result<ResolvedFamilyOptions>,
    pub(super) finish: fn(&mut ResidentModelBuildPlan) -> Result<()>,
}

pub(super) struct ResolvedStrategy {
    pub(super) execution: ResolvedExecution,
    pub(super) backend: ModelExecutionBackend,
    pub(super) catalog_backend: BackendSelection,
    pub(super) resolve: fn(&ModelDescriptor, BackendSelection) -> Result<ResolvedModelBackend>,
    pub(super) policy: &'static ModelOptionPolicy,
    pub(super) parallel: Option<PipelineBuildOptions>,
}

pub(super) const RESIDENT_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: common_serving_options,
    parallel: reject_parallel,
    adapt: generic_options,
    finish: no_provider_admission,
};

pub(super) fn no_provider_admission(_: &mut ResidentModelBuildPlan) -> Result<()> {
    Ok(())
}

pub(super) fn common_serving_options(
    _: &ModelFamily,
    serving: &ServingModelOptions,
    options: ModelFactoryOptions,
    construction: ResolvedConstruction,
) -> Result<ModelFactoryOptions> {
    serving.resolve_options(construction, options)
}
pub(super) fn reject_parallel(
    _: ResidentModelBuildPlan,
    _: PipelineBuildOptions,
    _: ResolvedExecution,
) -> Result<ResidentModelBuildPlan> {
    Err(invalid(
        "selected resident strategy does not accept parallel options",
    ))
}
pub(super) fn implementation(family: &ModelFamily) -> Result<&'static ModelImplementation> {
    MODEL_IMPLEMENTATIONS
        .iter()
        .find(|entry| &entry.family == family)
        .ok_or_else(|| {
            invalid(format!(
                "no model implementation can build family '{family}'"
            ))
        })
}
pub(super) fn select_resident_strategy(
    implementation: &ModelImplementation,
    backend: BackendSelection,
    request: &SelectionRequest,
    _: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    request.reject_specialized()?;
    if request.serving.parallel.is_some() {
        return Err(invalid(
            "this model family has no standard pipeline serving binding",
        ));
    }
    let backend =
        Option::<ModelExecutionBackend>::from(backend).unwrap_or(ModelExecutionBackend::Cuda);
    Ok(ResolvedStrategy {
        execution: ResolvedExecution::new(
            backend,
            ResolvedConstruction::Resident(implementation.build),
            StorageEncoding::ProfileDefined,
            ArithmeticPrecision::ModelDefined,
            "resident catalog/configuration validated; checkpoint binding not yet validated",
        ),
        backend,
        catalog_backend: backend.into(),
        resolve: implementation.resolve,
        policy: &RESIDENT_OPTIONS,
        parallel: None,
    })
}
// Exhaustive destructuring makes adding a request/launch field require updating
// this assertion. Omitted placement/launch is not an explicit default.
fn same_parallel_request(left: &PipelineBuildOptions, right: &PipelineBuildOptions) -> bool {
    let PipelineBuildOptions {
        parallelism,
        rank_backend,
        devices,
        rank_timeout,
        rank_restarts,
        #[cfg(unix)]
        process_launch,
    } = left;
    if parallelism != &right.parallelism
        || rank_backend != &right.rank_backend
        || devices != &right.devices
        || rank_timeout != &right.rank_timeout
        || rank_restarts != &right.rank_restarts
    {
        return false;
    }
    #[cfg(unix)]
    {
        use ferrule_runtime::parallel::process::ProcessLaunch;
        match (process_launch, &right.process_launch) {
            (None, None) => {}
            (
                Some(ProcessLaunch {
                    executable,
                    args,
                    environment,
                }),
                Some(other),
            ) if executable == &other.executable
                && args == &other.args
                && environment == &other.environment => {}
            _ => return false,
        }
    }
    true
}

fn validate_selected_parallel(
    backend: ModelExecutionBackend,
    policy: Option<execution_policy::TopologyPolicy>,
    parallel: Option<&PipelineBuildOptions>,
) -> Result<Option<execution_policy::ValidatedTopology>> {
    match (policy, parallel) {
        (Some(policy), parallel) => parallel
            .unwrap_or(&PipelineBuildOptions::default())
            .validate_policy(backend, policy.layout, policy.execution, policy.placement)
            .map(Some),
        (None, Some(_)) => Err(invalid(
            "selected strategy has no topology policy for parallel options",
        )),
        (None, None) => Ok(None),
    }
}

impl ResolvedStrategy {
    fn validate_parallel(&self, compatibility: Option<&PipelineBuildOptions>) -> Result<()> {
        if let Some(supplied) = compatibility
            && self
                .parallel
                .as_ref()
                .is_none_or(|selected| !same_parallel_request(selected, supplied))
        {
            return Err(invalid(
                "parallel options differ from the frozen selected strategy",
            ));
        }
        validate_selected_parallel(
            self.backend,
            self.execution.topology_policy,
            self.parallel.as_ref(),
        )?;
        Ok(())
    }

    pub(super) fn resolve_options(
        &self,
        family: &ModelFamily,
        serving: &ServingModelOptions,
        options: ModelFactoryOptions,
    ) -> Result<ModelFactoryOptions> {
        self.validate_parallel(serving.parallel.as_ref())?;
        (self.policy.serving)(family, serving, options, self.execution.construction)
    }
    pub(super) fn attach(
        mut self,
        plan: ResidentModelBuildPlan,
        parallel: PipelineBuildOptions,
    ) -> Result<ResidentModelBuildPlan> {
        self.validate_parallel(Some(&parallel))?;
        let parallel = self
            .parallel
            .take()
            .expect("validated selected parallel request");
        let mut execution = self.execution;
        execution.option_policy = self.policy;
        (self.policy.parallel)(plan, parallel, execution)
    }
}
impl ResidentModelPlanner {
    pub(super) fn prepare_strategy(
        &self,
        config: &AutoConfig,
        requested_backend: BackendSelection,
        chat_template: Option<&str>,
        options: ModelFactoryOptions,
        parallel: Option<PipelineBuildOptions>,
        mut strategy: ResolvedStrategy,
    ) -> Result<ResidentModelBuildPlan> {
        // Legacy callers may repeat the request, but may not replace the selector's
        // request. Validate before any resolver or model option/admission hook.
        strategy.validate_parallel(parallel.as_ref())?;
        strategy.execution.option_policy = strategy.policy;
        let entry = (strategy.resolve)(config.descriptor(), strategy.catalog_backend)?;
        let mut plan = self.prepare_with_entry(
            config,
            requested_backend,
            chat_template,
            options,
            Some(strategy.execution),
            entry,
        )?;
        plan.requested_backend = requested_backend;
        plan.backend = strategy.backend;
        plan.request.backend = strategy.backend;
        plan.request.model_info = ferrule_model::ModelInfo::from_descriptor(
            config.descriptor(),
            strategy.backend.as_str(),
        );
        if let Some(parallel) = strategy.parallel.clone() {
            plan = strategy.attach(plan, parallel)?;
        }
        plan.refresh_facts()?;
        Ok(plan)
    }

    pub(super) fn prepare_serving_request(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template: Option<&str>,
        options: ModelFactoryOptions,
        request: SelectionRequest,
    ) -> Result<ResidentModelBuildPlan> {
        let request = request.normalize(backend, &options)?;
        let implementation = implementation(&config.descriptor().spec.family)?;
        self.prepare_catalog_serving(
            implementation,
            config,
            backend,
            chat_template,
            options,
            request,
        )
    }

    pub(super) fn prepare_catalog_serving(
        &self,
        implementation: &ModelImplementation,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template: Option<&str>,
        options: ModelFactoryOptions,
        request: SelectionRequest,
    ) -> Result<ResidentModelBuildPlan> {
        let mut strategy = (implementation.select)(implementation, backend, &request, &options)?;
        strategy.execution.expert_read_pipeline = request.serving.expert_read_pipeline;
        if request.serving.model_precision == Some(ModelPrecision::F32)
            && !matches!(
                strategy.execution.support.arithmetic,
                ArithmeticPrecision::F32
                    | ArithmeticPrecision::F32Tf32x3
                    | ArithmeticPrecision::W8A16Bf16F32Accumulate
            )
        {
            return Err(invalid(
                "explicit F32 model precision is not implemented by this profile; omit model_precision to preserve its legacy defaults",
            ));
        }
        let options = strategy.resolve_options(
            &config.descriptor().spec.family,
            &request.serving,
            options,
        )?;
        self.prepare_strategy(config, backend, chat_template, options, None, strategy)
    }
}

impl ResidentModelBuildPlan {
    pub(super) fn refresh_facts(&mut self) -> Result<()> {
        let requested = self.requested_parallel.as_ref();
        let attached = self.request.pipeline.as_ref();
        if let (Some(requested), Some(attached)) = (requested, attached)
            && !same_parallel_request(requested, attached)
        {
            return Err(invalid(
                "attached parallel options differ from the frozen selected request",
            ));
        }
        let options = requested.or(attached);
        let validated = validate_selected_parallel(
            self.backend,
            self.request.execution.topology_policy,
            options,
        )?;
        let mut topology = resident_topology();
        if let (Some(policy), Some(options), Some(validated)) =
            (self.request.execution.topology_policy, options, validated)
        {
            topology.rank_backend = Some(options.rank_backend);
            topology.logical = logical_topology(options);
            if self.backend == ModelExecutionBackend::Cuda {
                let placement = execution_policy::validate_placement(
                    validated,
                    execution_policy::PlacementTarget::Device,
                    options.devices.as_deref(),
                    policy.placement,
                )
                .map_err(|e| invalid(e.to_string()))?;
                let devices = placement
                    .requested_devices()
                    .map(<[usize]>::to_vec)
                    .unwrap_or_else(|| {
                        (0..validated.device_slots())
                            .map(|slot| placement.device(slot).expect("validated slot"))
                            .collect()
                    });
                if validated.layout() == execution_policy::OwnerLayout::RootWithExperts {
                    topology.physical.root_device = devices.first().copied();
                    topology.physical.expert_devices = Some(devices.clone());
                    topology.physical.source_identity = Some(ParallelRankId::new(0));
                }
                topology.physical.devices = Some(devices);
            }
        } else if self.backend == ModelExecutionBackend::Cuda {
            topology.physical.root_device = Some(0);
            topology.physical.devices = Some(vec![0]);
        }
        expert_read_policy::validate_resolved(&self.request)?;
        self.topology = topology;
        self.provider_adjustments = AdjustmentReport::default();
        (self.request.execution.option_policy.finish)(self)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/catalog_policy.rs"]
mod catalog_policy_tests;

pub(super) fn apply_common_serving_options(
    serving: &ServingModelOptions,
    options: &mut ModelFactoryOptions,
) {
    if let Some(n) = serving.moe_hotset_experts {
        options.moe_hotset_experts = n;
    }
    if let Some(n) = serving.max_tensor_mebibytes {
        options.max_tensor_mebibytes = n;
    }
}

pub(super) fn validate_serving_engine(
    serving: &ServingModelOptions,
    construction: ResolvedConstruction,
) -> Result<()> {
    if serving.engine == ServingEngineSelection::Resident
        && construction.kind() == SelectedImplementationKind::Pipeline
    {
        return Err(invalid(
            "parallel/rank options require --engine pipeline (or auto)",
        ));
    }
    Ok(())
}

impl ResidentModelBuildPlan {
    /// Metadata-only effective context limits. Shape-dependent memory stays
    /// unknown until the owner resolves KV and model memory; legacy requested,
    /// effective and resolution reports are intentionally unchanged.
    pub fn context_capacity_profile(
        &self,
    ) -> Result<ferrule_runtime::engine::ContextCapacityProfile> {
        ferrule_runtime::engine::ContextCapacityProfile::from_configs(
            self.request.scheduler_config,
            self.request.driver_config,
        )
    }
}

pub(super) fn same_final_request(
    left: &ResolvedModelRequest,
    right: &ResolvedModelRequest,
) -> bool {
    // Exhaustive destructuring requires new fields to participate in freezing.
    let ResolvedModelRequest {
        execution,
        family,
        backend,
        model_path,
        model_info,
        pipeline,
        family_options,
        max_layers,
        max_tensor_bytes,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes,
        expert_memory_policy,
        moe_hotset_experts,
        kv_cache_bytes,
        scheduler_config,
        driver_config,
    } = left;
    let same_execution = std::ptr::eq(execution.option_policy, right.execution.option_policy)
        && std::ptr::fn_addr_eq(
            execution.construction.builder(),
            right.execution.construction.builder(),
        )
        && execution.construction.kind() == right.execution.construction.kind()
        && execution.expert_read_pipeline == right.execution.expert_read_pipeline
        && execution.numeric_bf16_precision == right.execution.numeric_bf16_precision
        && execution.numeric_profile == right.execution.numeric_profile
        && execution.topology_policy == right.execution.topology_policy
        && match (execution.conversion, right.execution.conversion) {
            (PipelineConversion::Allowed, PipelineConversion::Allowed) => true,
            (PipelineConversion::Frozen(a), PipelineConversion::Frozen(b)) => a == b,
            _ => false,
        }
        && execution.adjustment_reason == right.execution.adjustment_reason
        && execution.kv_execution_policy == right.execution.kv_execution_policy
        && execution.support == right.execution.support;
    let same_family = match (family_options, &right.family_options) {
        (ResolvedFamilyOptions::Qwen35(a), ResolvedFamilyOptions::Qwen35(b)) => {
            (match (a.moe(), b.moe()) {
                (Some(a), Some(b)) => {
                    a.device_capacity == b.device_capacity
                        && a.numeric_profile == b.numeric_profile
                        && a.host_cache == b.host_cache
                }
                (None, None) => true,
                _ => false,
            }) && a.expert_placement() == b.expert_placement()
        }
        (ResolvedFamilyOptions::Generic, ResolvedFamilyOptions::Generic) => true,
        _ => false,
    };
    let same_model = model_info.family == right.model_info.family
        && model_info.architecture == right.model_info.architecture
        && model_info.attention == right.model_info.attention
        && model_info.weight_source == right.model_info.weight_source
        && model_info.hidden_size == right.model_info.hidden_size
        && model_info.num_layers == right.model_info.num_layers
        && model_info.num_experts == right.model_info.num_experts
        && model_info.num_experts_per_tok == right.model_info.num_experts_per_tok
        && model_info.vocab_size == right.model_info.vocab_size
        && model_info.backend == right.model_info.backend;
    let same_pipeline = match (pipeline, &right.pipeline) {
        (Some(a), Some(b)) => {
            a.parallelism == b.parallelism
                && a.rank_backend == b.rank_backend
                && a.devices == b.devices
                && a.rank_timeout == b.rank_timeout
                && a.rank_restarts == b.rank_restarts
                && {
                    #[cfg(unix)]
                    {
                        match (&a.process_launch, &b.process_launch) {
                            (None, None) => true,
                            (Some(a), Some(b)) => {
                                a.executable == b.executable
                                    && a.args == b.args
                                    && a.environment == b.environment
                            }
                            _ => false,
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        true
                    }
                }
        }
        (None, None) => true,
        _ => false,
    };
    same_execution
        && same_family
        && same_model
        && same_pipeline
        && family == &right.family
        && backend == &right.backend
        && model_path == &right.model_path
        && max_layers == &right.max_layers
        && max_tensor_bytes == &right.max_tensor_bytes
        && output_head_chunk_rows == &right.output_head_chunk_rows
        && expert_reader_max_tensor_bytes == &right.expert_reader_max_tensor_bytes
        && expert_memory_policy == &right.expert_memory_policy
        && moe_hotset_experts == &right.moe_hotset_experts
        && kv_cache_bytes == &right.kv_cache_bytes
        && scheduler_config == &right.scheduler_config
        && driver_config.ctx_size == right.driver_config.ctx_size
        && driver_config.stop_at_eos == right.driver_config.stop_at_eos
        && driver_config.enable_native_proposals == right.driver_config.enable_native_proposals
        && driver_config.proposal_confidence_threshold.to_bits()
            == right.driver_config.proposal_confidence_threshold.to_bits()
}
