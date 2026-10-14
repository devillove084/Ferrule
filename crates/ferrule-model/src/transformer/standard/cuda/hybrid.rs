//! CUDA protocol adapter over the one standard hybrid layer/output composition.
use super::super::{PreparedStandardLayer, PreparedStandardOutput, StandardTransformerHidden};

use super::super::RoutedLayerExecution;
use super::*;
use crate::decoder::{
    CudaHybridSequenceState, CudaKvView, DecoderLogits, DecoderTransactionContext,
    HybridCudaDevice, HybridCudaRoutedExperts, HybridLayerSchema, HybridStateSchema,
    NoTerminalGuard, PackedDecoderBatch, StandardSequenceState,
};
use crate::transformer::{
    Attention, BoundDecoderResources, LayerRequest, MemoryLayerWeightCache, NoPostLayerTap, Poll,
    PreparedTransformer, StateDictMaterializer, Step, TransformerModule,
};
use std::convert::Infallible;

#[path = "invocation_trace.rs"]
mod invocation_trace;
use invocation_trace::InvocationTrace;

#[path = "external_experts.rs"]
pub(super) mod external_experts;
#[cfg(test)]
use external_experts::{ExpertForward, ExpertForwardPhase};
use external_experts::{ExternalExpertCustody, RoutedCall, routed_sequences};

pub struct CudaHybridModule {
    resources: Arc<BoundDecoderResources>,
    materializer: Arc<StateDictMaterializer>,
    device: HybridCudaDevice,
    schema: HybridStateSchema,
    attention_capacity: Option<crate::decoder::mixed_attention::MixedAttentionCapacity>,
    external: ExternalExpertCustody,
    closed: bool,
    performance: Option<InvocationTrace>,
    #[cfg(test)]
    reject_next_finish_completion: std::cell::Cell<bool>,
    operators: CudaStandardDecoderOperators,
    prepared: PreparedTransformer<
        PreparedEmbedding,
        PreparedStandardLayer,
        PreparedStandardOutput,
        NoPostLayerTap,
    >,
}
impl std::fmt::Debug for CudaHybridModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHybridModule")
            .field("device", &self.device)
            .field("operators", &self.operators)
            .field("external", &self.external)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}
impl CudaHybridModule {
    pub fn prepare(
        resources: Arc<BoundDecoderResources>,
        device: HybridCudaDevice,
        options: &crate::decoder::GenericDecoderOptions,
    ) -> Result<Self> {
        Self::prepare_with_routed_experts(resources, device, options, None)
    }

    pub fn prepare_with_routed_experts(
        resources: Arc<BoundDecoderResources>,
        device: HybridCudaDevice,
        options: &crate::decoder::GenericDecoderOptions,
        mut experts: Option<HybridCudaRoutedExperts>,
    ) -> Result<Self> {
        if let Some(experts) = &mut experts {
            experts.owner = Some(device.clone());
        }
        let estimate = if experts.is_some() {
            crate::decoder::HybridCudaMemoryEstimate::for_resources_with_routed_experts
        } else {
            crate::decoder::HybridCudaMemoryEstimate::for_resources
        }(&resources, options, device.budget().kv_pages)?;
        let claim = device.admit(estimate, options.capabilities().max_sequences)?;
        let result = Self::prepare_with_claim(resources, device, options, experts, &claim);
        claim.finish(result)
    }

    /// Prepare under an outer factory claim; pool/default-state construction
    /// remains inside that factory's rollback scope. Does not finish the claim.
    pub fn prepare_with_claim(
        resources: Arc<BoundDecoderResources>,
        device: HybridCudaDevice,
        options: &crate::decoder::GenericDecoderOptions,
        mut experts: Option<HybridCudaRoutedExperts>,
        claim: &crate::decoder::HybridCudaPreparationClaim,
    ) -> Result<Self> {
        if let Some(experts) = &mut experts {
            experts.owner = Some(device.clone());
        }
        let estimate = if experts.is_some() {
            crate::decoder::HybridCudaMemoryEstimate::for_resources_with_routed_experts
        } else {
            crate::decoder::HybridCudaMemoryEstimate::for_resources
        }(&resources, options, device.budget().kv_pages)?;
        claim.validate_estimate(estimate, options.capabilities().max_sequences)?;
        claim.begin_image(&device)?;
        options.validate_hybrid_cuda(&resources)?;
        let attention_capacity =
            crate::decoder::mixed_attention::MixedAttentionCapacity::for_resources(
                &resources, options,
            )?;
        let spec = resources.spec();
        let active_layers = options.active_layers().unwrap_or(spec.layers().len());

        let schema = HybridStateSchema::from_spec(spec, active_layers)?;
        super::super::validate_hybrid_descriptors(spec, spec.layers())?;
        for layer in spec.layers() {
            if let Attention::Gqa(g) = layer.attention() {
                validate_rope(g.rotary())?;
            }
        }

        crate::moe::catalog::ValidatedExpertCatalog::for_decoder_range(
            &resources,
            0..active_layers,
        )?;
        device.ensure_ready()?;
        let materializer = Arc::new(StateDictMaterializer::new(options.max_parameter_bytes())?);
        let embedding = super::super::prepare_embedding(&resources, &materializer)?;
        let output = super::super::prepare_output(&resources, &materializer)?;
        let mut layers = Vec::new();
        for layer in &spec.layers()[..active_layers] {
            layers.push(super::super::hybrid::prepare_standard_layer(
                layer,
                &resources,
                &materializer,
                &mut MemoryLayerWeightCache::new(),
                options.max_positions(),
            )?);
        }
        let mut operators = match options.hybrid_cuda_numeric_fp8_config() {
            Some((limits, scratch, precision)) => {
                CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                    Rc::clone(device.operators()),
                    resources.state_dict().parameters(),
                    ExpertCachePolicy::Bounded(limits),
                    scratch,
                    precision,
                )?
            }
            None => CudaStandardDecoderOperators::new(
                Rc::clone(device.operators()),
                options.precision(),
                resources.state_dict().parameters(),
            )?,
        };
        let chunk_bytes =
            crate::decoder::HybridMemoryPlan::bf16_chunk_workspace_bytes(&resources, options)?;
        let gemmex_plans =
            crate::decoder::HybridMemoryPlan::bf16_gemmex_plans(&resources, options)?;
        let result = operators
            .prepare_bf16_gemmex(&gemmex_plans)
            .and_then(|()| {
                operators.prepare_bf16_attention(
                    crate::decoder::HybridMemoryPlan::bf16_attention_plan(&resources, options)?,
                )
            })
            .and_then(|()| operators.prepare_bf16_chunk_workspace(chunk_bytes))
            .and_then(|()| {
                operators.prepare_hybrid_image(&embedding, &layers, &output, &materializer)
            });
        if operators.needs_quarantine() {
            device.quarantine();
        }
        result?;
        if operators.resident_parameter_bytes() > device.budget().weight_bytes {
            return Err(unsupported("resident CUDA weights exceed budget"));
        }
        let image = Self {
            resources,
            materializer,
            external: ExternalExpertCustody::new(experts, &device),
            device,
            schema,
            attention_capacity,
            closed: false,
            performance: None,
            #[cfg(test)]
            reject_next_finish_completion: std::cell::Cell::new(false),
            operators,
            prepared: PreparedTransformer::new(embedding, layers, output, NoPostLayerTap),
        };
        claim.image_prepared()?;
        Ok(image)
    }
    #[cfg(test)]
    pub(crate) fn reject_next_finish_completion(&self) {
        self.reject_next_finish_completion.set(true);
    }

    pub fn set_diagnostic_trace<F>(&self, callback: F)
    where
        F: FnMut(&CudaDiagnosticEvent<'_>) -> Result<()> + 'static,
    {
        self.operators.set_diagnostic_trace(callback);
    }
    pub fn clear_diagnostic_trace(&self) {
        self.operators.clear_diagnostic_trace();
    }

    /// Immutable physical/native profile, independent of numeric FP8 arithmetic.
    pub fn bf16_profile(&self) -> Option<crate::transformer::Bf16HybridCudaProfile> {
        self.operators
            .bf16
            .as_ref()
            .filter(|_| self.operators.numeric.is_none())
            .map(|_| crate::transformer::Bf16HybridCudaProfile)
    }
    /// Shared dtype/storage contract. Use `numeric_fp8_precision()` for the
    /// distinct fused versus tiled execution identity.
    pub fn numeric_bf16_profile(
        &self,
    ) -> Option<crate::transformer::NumericFp8Bf16HybridCudaProfile> {
        self.numeric_fp8_precision()
            .is_some_and(crate::transformer::NumericFp8Precision::has_bf16_model_boundaries)
            .then_some(crate::transformer::NumericFp8Bf16HybridCudaProfile)
    }
    pub fn resident_parameter_bytes(&self) -> usize {
        self.operators.resident_parameter_bytes()
    }
    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.operators.expert_cache_stats()
    }
    pub fn expert_metadata_preflight_stats(&self) -> ExpertMetadataPreflightStats {
        self.operators.expert_metadata_preflight_stats()
    }
    pub fn numeric_fp8_precision(&self) -> Option<crate::transformer::NumericFp8Precision> {
        self.operators.numeric_fp8_precision()
    }
    /// Logical image-owned GDN chunk scratch; shared by layers, never sequence state.
    pub fn bf16_chunk_workspace_bytes(&self) -> usize {
        self.operators.bf16_chunk_workspace_bytes()
    }

    /// Reservation, actual workspace payload, allocation count, reuse count, submissions.
    pub fn numeric_workspace_usage(&self) -> Option<(usize, usize, u64, u64, u64)> {
        self.operators.numeric_workspace_stats().map(|s| {
            (
                s.reserved_bytes,
                s.allocated_bytes,
                s.allocations,
                s.reuses,
                s.submissions,
            )
        })
    }
    pub fn spec(&self) -> &crate::transformer::DecoderModelSpec {
        self.resources.spec()
    }
    fn check_expert_active(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        self.external.check(&self.device, transaction, batch)
    }

    fn expert_failed(&mut self) -> Result<()> {
        self.external.fail()
    }

    fn drain_experts(&mut self, shutdown: bool) -> Result<bool> {
        self.external.drain(&self.device, shutdown)
    }

    fn require_expert_drain(&mut self, shutdown: bool) -> Result<()> {
        self.external.require_drain(&self.device, shutdown)
    }

    fn checked<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if let Err(error) = self.device.ensure_ready() {
            self.end_performance("failed");
            return Err(error);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = run(self);
            result.map_err(|primary| {
                Error::with_cleanup(
                    "CUDA hybrid expert cancellation",
                    primary,
                    self.expert_failed(),
                )
            })
        }));
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        match result {
            Ok(result) => {
                if result.is_err() {
                    self.end_performance("failed");
                }
                result
            }
            Err(panic) => {
                self.device.quarantine();
                self.operators.poisoned = true;
                self.end_performance("panicked");
                std::panic::resume_unwind(panic)
            }
        }
    }
    fn execute_layer(
        &mut self,
        context: &DecoderTransactionContext,
        index: usize,
        batch: &PackedDecoderBatch,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        hidden: &mut StandardTransformerHidden,
    ) -> Result<()> {
        self.checked(|this| {
            this.operators.set_diagnostic_layer(Some(index));
            let layer = this
                .prepared
                .layers()
                .get(index)
                .ok_or_else(|| cuda_error("missing CUDA hybrid layer"))?;
            let kv_layer = match this.schema.layers()[index] {
                HybridLayerSchema::FullAttention { kv_layer, .. } => kv_layer,
                _ => 0,
            };
            if context.transaction() != kv.transaction() {
                return Err(cuda_error("external expert/root KV transaction mismatch"));
            }
            batch.validate_states(states)?;
            let row_sequences = routed_sequences(batch)?;
            let forward = this.external.forward.as_ref();
            let device = &this.device;
            let (source_rank, executor, mut external_check) =
                match this.external.attachment.as_mut() {
                    Some(experts) => (
                        experts.source_rank,
                        Some(experts.executor.as_mut().expect("attached executor")),
                        experts.check_active.as_mut(),
                    ),
                    None => (ParallelRankId::new(0), None, None),
                };
            let mut check_active = |transaction| {
                device.ensure_ready()?;
                forward
                    .ok_or_else(|| cuda_error("unknown external expert forward"))?
                    .check(transaction, batch)?;
                if let Some(check) = &mut external_check {
                    check(transaction)?;
                }
                Ok(())
            };
            let mut call = executor.map(|executor| RoutedCall {
                executor: executor.as_mut(),
                device,
                holds: &mut this.external.holds,
            });
            let mut execution = call.as_mut().map(|executor| RoutedLayerExecution {
                transaction: context.transaction(),
                source_rank,
                sequences: &row_sequences,
                executor,
                check_active: &mut check_active,
            });
            if let Some(execution) = &mut execution {
                execution.check_active()?;
            }
            let mut states = states
                .iter_mut()
                .map(|s| s as &mut dyn StandardSequenceState)
                .collect::<Vec<_>>();
            let attention = this.operators.bf16_attention.clone();
            let result = CudaStandardKvBinding::with_context(
                kv,
                batch,
                context,
                kv_layer,
                attention,
                |binding| {
                    this.operators.execute_root_layer(
                        &this.materializer,
                        layer.index(),
                        layer.attention().block().as_ref(),
                        layer.feed_forward().block(),
                        kv_layer,
                        hidden,
                        binding,
                        &mut states,
                        execution.as_mut(),
                    )
                },
            );
            this.operators.set_diagnostic_layer(None);
            result?;
            this.require_expert_drain(false)
        })
    }
}
fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_cuda", message)),
    }
}

impl TransformerModule for CudaHybridModule {
    type State = CudaHybridSequenceState;
    type KvView = CudaKvView;
    type Hidden = StandardTransformerHidden;
    type ArenaKey = (usize, usize);
    type Arena = ();
    type EmbeddingPending = Infallible;
    type LayerPending = Infallible;
    type OutputPending = Infallible;
    type Quiescence = ();
    type Event = Infallible;
    type TerminalGuard = NoTerminalGuard;

    fn physical_resource_snapshot(
        &self,
        default_state: &Self::State,
    ) -> crate::resource_observability::PhysicalResourceObservation {
        use crate::resource_observability::{
            DefaultStateObservation, ExpertCacheObservation, PhysicalResourceObservation,
            ResourceClosure, ResourceEvidence, ResourceQuiescence, ResourceResidency,
            TransferObservation, WorkspaceObservation,
        };
        let image_generation = self.operators.image_generation();
        let transfer = self.operators.boundary.observation_stats();
        let quarantined = self.device.needs_quarantine()
            || self.operators.needs_quarantine()
            || default_state.kv_state().needs_quarantine();
        let mut snapshot = PhysicalResourceObservation {
            image_generation: ResourceEvidence::Observed(image_generation),
            device_ordinal: ResourceEvidence::Observed(self.device.device_ordinal()),
            default_state: ResourceEvidence::Observed(DefaultStateObservation {
                topology_id: default_state.topology_id().get(),
                generation: default_state.core().generation(),
                position: default_state.core().position(),
                poisoned: default_state.core().is_poisoned(),
                resident_bytes: ResourceEvidence::Observed(
                    default_state.kv_state().resident_bytes(),
                ),
            }),
            resident_parameter_bytes: ResourceEvidence::Observed(self.resident_parameter_bytes()),
            live_model_state_bytes: ResourceEvidence::Observed(self.device.live_state_bytes()),
            expert_cache: self
                .expert_cache_stats()
                .map(|stats| ResourceEvidence::Observed(ExpertCacheObservation::from(stats)))
                .unwrap_or_default(),
            numeric_workspace: self
                .operators
                .numeric_workspace_stats()
                .map(|stats| {
                    ResourceEvidence::Observed(WorkspaceObservation {
                        image_generation,
                        kind: "numeric",
                        reserved_bytes: stats.reserved_bytes,
                        allocated_bytes: stats.allocated_bytes,
                        creations: stats.allocations,
                        reuses: stats.reuses,
                        submissions: stats.submissions,
                    })
                })
                .unwrap_or_default(),
            recurrent_chunk_workspace_bytes: ResourceEvidence::Observed(
                self.bf16_chunk_workspace_bytes(),
            ),
            boundary_transfer: ResourceEvidence::Observed(TransferObservation {
                tx_slots: transfer.tx_slots,
                rx_slots: transfer.rx_slots,
                tx_in_use: transfer.tx_in_use,
                rx_in_use: transfer.rx_in_use,
                tx_high_water: transfer.tx_high_water,
                rx_high_water: transfer.rx_high_water,
                pinned_allocations: transfer.pinned_allocations,
                pinned_bytes: transfer.pinned_bytes,
                device_holds_in_use: transfer.device_holds_in_use,
                device_holds_high_water: transfer.device_holds_high_water,
                submissions: transfer.submissions,
                failed_submissions: transfer.failed_submissions,
                completions: transfer.completions,
                cancel_intents: transfer.cancel_intents,
                canceled_completions: transfer.canceled_completions,
                quarantined: transfer.quarantined,
            }),
            // allocator_metrics polls retirement fences and can release blocks.
            // No immutable allocator getter is exposed; leave it Unobserved.
            allocator: ResourceEvidence::Unobserved,
            owner_fence_completions: ResourceEvidence::Observed(
                self.device.completed_state_operations(),
            ),
            quarantine: ResourceEvidence::Observed(quarantined),
            residency: ResourceEvidence::Observed(ResourceResidency::Retained),
            close: ResourceEvidence::Observed(if self.closed {
                ResourceClosure::Closed
            } else {
                ResourceClosure::Open
            }),
            ..Default::default()
        };
        if quarantined {
            snapshot.mark_unknown();
        } else if self.closed {
            // closed is published only after external drain, operator quiescence
            // and both device fences succeed in the existing shutdown protocol.
            snapshot.quiescence = ResourceEvidence::Verified(ResourceQuiescence::Quiescent);
        }
        // The live image, default state and workspaces still own their storage.
        // Neither a successful close nor allocator zeros proves post-Drop release.
        snapshot
    }

    fn layer_count(&self) -> usize {
        self.prepared.layers().len()
    }

    fn validate(
        &mut self,
        context: &crate::decoder::DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
    ) -> Result<()> {
        self.begin_performance(context.transaction(), batch);
        let result = (|| {
            self.device.ensure_ready()?;
            if let Some(append) = context.invocation() {
                append.cuda_host_kv()?;
                append.validate_batch(context.transaction(), batch)?;
                if append.layer_count() != self.prepared.layers().len()
                    || self
                        .schema
                        .layers()
                        .iter()
                        .any(|l| !matches!(l, HybridLayerSchema::FullAttention { .. }))
                    || self.external.attachment.is_some()
                    || self.operators.numeric.is_some()
                    || self.operators.bf16.is_some()
                    || self.operators.tensor_collective.is_some()
                {
                    return Err(cuda_error(
                        "HostKV decode requires dense F32 all-GQA layers; no numeric/EP/TP fallback",
                    ));
                }
            }
            if self.closed || self.external.forward.is_some() {
                return Err(cuda_error(
                    "hybrid module closed or previous expert forward retains custody",
                ));
            }
            if context.transaction() != kv.transaction() {
                return Err(cuda_error("hybrid context/root KV transaction mismatch"));
            }
            batch.validate_states(states)?;
            if let Some(capacity) = self.attention_capacity {
                capacity.validate_batch(batch)?;
            }
            routed_sequences(batch)?;
            self.operators.preflight_expert_metadata()?;
            for state in states {
                state.validate_standard_state(&self.schema)?;
            }
            kv.validate_batch(batch)?;
            if let Some(owner) = &self.operators.bf16_attention {
                let owner = owner.borrow();
                for layer in self.schema.layers() {
                    if let HybridLayerSchema::FullAttention { kv_layer, .. } = layer {
                        kv.with_bf16_gqa_planes(&self.operators.ops, *kv_layer, batch, |planes| {
                            planes
                                .prepared_attention_view(owner.provider.plan())
                                .map(|_| ())
                        })?;
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.end_performance("failed");
        }
        result
    }

    fn arena_key(&self, batch: &PackedDecoderBatch) -> Result<Self::ArenaKey> {
        Ok((batch.len(), batch.sequences().len()))
    }

    fn build_arena(&mut self, _key: &Self::ArenaKey) -> Result<Self::Arena> {
        Ok(())
    }

    fn submit_embedding(
        &mut self,
        context: &crate::decoder::DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
    ) -> Result<Step<Self::Hidden, Self::EmbeddingPending>> {
        self.checked(|this| {
            this.external.begin(context.transaction(), batch, kv)?;
            this.operators.begin_bf16_attention(context.transaction())?;
            let metadata = super::super::packed_gqa_metadata(batch)?;
            let rows = super::super::ready(this.operators.embedding(
                this.prepared.embedding(),
                batch.token_ids(),
                None,
            )?)?;
            Ok(Step::Complete(StandardTransformerHidden {
                rows: Some(rows),
                metadata,
            }))
        })
    }

    fn poll_embedding(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
        pending: &mut Self::EmbeddingPending,
    ) -> Result<Poll<Self::Hidden, Self::EmbeddingPending>> {
        match *pending {}
    }

    fn start_layer(
        &mut self,
        context: &crate::decoder::DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, Self::State, Self::KvView>,
        hidden: &mut Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<Step<Vec<Self::Event>, Self::LayerPending>> {
        let LayerRequest::Target { batch, kv, states } = request else {
            return Err(cuda_error("CUDA hybrid module has no proposal stages"));
        };
        self.execute_layer(context, layer, batch, states, kv, hidden)?;
        Ok(Step::Complete(Vec::new()))
    }

    fn poll_layer(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _layer: usize,
        _request: LayerRequest<'_, Self::State, Self::KvView>,
        _hidden: &mut Self::Hidden,
        _arena: &mut Self::Arena,
        pending: &mut Self::LayerPending,
    ) -> Result<Poll<Vec<Self::Event>, Self::LayerPending>> {
        match *pending {}
    }

    fn post_layer(
        &mut self,
        _layer: usize,
        _batch: &PackedDecoderBatch,
        _hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<()> {
        Ok(())
    }

    fn submit_output(
        &mut self,
        context: &crate::decoder::DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<Step<DecoderLogits, Self::OutputPending>> {
        self.checked(|this| {
            this.check_expert_active(context.transaction(), batch)?;
            this.operators
                .finish_bf16_attention(context.transaction(), false)?;
            let logits = super::super::execute_standard_output(
                &mut this.operators,
                this.prepared.output(),
                hidden,
            )?;
            this.check_expert_active(context.transaction(), batch)?;
            Ok(Step::Complete(DecoderLogits::Dense(logits)))
        })
    }

    fn poll_output(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
        pending: &mut Self::OutputPending,
    ) -> Result<Poll<DecoderLogits, Self::OutputPending>> {
        match *pending {}
    }

    fn poll_embedding_cancel(&mut self, pending: &mut Self::EmbeddingPending) -> Result<bool> {
        match *pending {}
    }

    fn poll_layer_cancel(
        &mut self,
        _layer: usize,
        pending: &mut Self::LayerPending,
    ) -> Result<bool> {
        match *pending {}
    }

    fn poll_output_cancel(&mut self, pending: &mut Self::OutputPending) -> Result<bool> {
        match *pending {}
    }

    fn cancel_embedding(&mut self, pending: Self::EmbeddingPending) -> Result<()> {
        match pending {}
    }

    fn cancel_layer(&mut self, _layer: usize, pending: Self::LayerPending) -> Result<()> {
        match pending {}
    }

    fn cancel_output(&mut self, pending: Self::OutputPending) -> Result<()> {
        match pending {}
    }

    fn begin_quiescence(&mut self) -> Result<Self::Quiescence> {
        self.checked(|this| {
            this.external.begin_drain()?;
            Ok(())
        })
    }

    fn poll_quiescence(&mut self, _quiescence: &mut Self::Quiescence) -> Result<bool> {
        self.checked(|this| {
            if !this.drain_experts(false)? {
                return Ok(false);
            }
            this.operators.quiesce()?;
            this.device.fence()?;
            Ok(true)
        })
    }

    fn finish(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
        events: Vec<Self::Event>,
    ) -> Result<Self::TerminalGuard> {
        let result = (|| {
            self.checked(|this| {
                this.check_expert_active(kv.transaction(), batch)?;
                this.require_expert_drain(false)?;
                this.check_expert_active(kv.transaction(), batch)
            })?;
            // Simulate lost completion evidence, not an illegal GPU operation. The
            // real fence then returns the owner's permanent typed Unknown result.
            #[cfg(test)]
            if self.reject_next_finish_completion.replace(false) {
                self.device.quarantine();
            }
            self.device.fence()?;
            for sequence in batch.sequences() {
                let state = states
                    .get(sequence.state_index())
                    .ok_or_else(|| cuda_error("missing hybrid sequence at completion"))?;
                state.validate_standard_state_at(&self.schema, sequence.sequence_len())?;
            }
            self.external.finish();
            match events.into_iter().next() {
                Some(event) => match event {},
                None => Ok(NoTerminalGuard),
            }
        })();
        self.end_performance(if result.is_ok() {
            "completed"
        } else {
            "failed"
        });
        result
    }

    fn abort(
        &mut self,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
    ) -> Result<()> {
        let result = self.checked(|this| {
            this.operators
                .finish_bf16_attention(kv.transaction(), true)?;
            this.require_expert_drain(false)?;
            this.device.fence()?;
            this.external.finish();
            Ok(())
        });
        self.end_performance(if result.is_ok() { "aborted" } else { "failed" });
        result
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.closed {
            return self.device.ensure_ready();
        }
        let result = self.checked(|this| {
            this.require_expert_drain(true)?;
            this.operators.quiesce()?;
            this.device.fence()?;
            this.external.close();
            this.closed = true;
            Ok(())
        });
        self.end_performance(if std::thread::panicking() {
            "panicked"
        } else if result.is_ok() {
            "shutdown"
        } else {
            "failed"
        });
        result
    }
}

impl Drop for CudaHybridModule {
    fn drop(&mut self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.shutdown()));
        if !matches!(result, Ok(Ok(()))) || self.device.needs_quarantine() {
            self.device.quarantine();
            self.operators.poisoned = true;
            self.external.retain_unknown();
            std::mem::forget(self.device.clone());
        }
    }
}

#[cfg(test)]
mod external_identity_tests {
    use super::*;
    use crate::decoder::{DecoderKvPageStatus, GenericDecoderSequenceState};
    use ferrule_common::execution::*;
    use std::num::NonZeroU32;

    fn packed(
        state: &GenericDecoderSequenceState,
        phase: ForwardPhase,
        generation: u64,
    ) -> PackedDecoderBatch {
        let batch = ExecutionBatch::new(
            match phase {
                ForwardPhase::Prefill => ForwardMode::Prefill,
                ForwardPhase::Decode => ForwardMode::Decode,
            },
            vec![1],
            vec![0],
            vec![Some(KvWriteSlot::new(2))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                phase,
                0..1,
                0,
                1,
                0..1,
            )],
            vec![KvBlockId::new(1)],
        );
        PackedDecoderBatch::lower(
            &batch,
            &[KvReservationView {
                state_slot: StateSlot::new(1),
                execution_state_slot: StateSlot::new(0),
                positions: 0..1,
                newly_allocated: vec![KvPageId(1)],
                generation,
                execution_generation: state.core().generation(),
                cow_replacement: None,
            }],
            std::slice::from_ref(state),
            &ExecutionCapabilities {
                max_batch_tokens: 2,
                max_sequences: 1,
                max_prefill_query_tokens_per_sequence: 2,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: NonZeroU32::new(2),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: NonZeroU32::new(2),
                kv_binding_mode: KvBindingMode::Paged,
                logits_row_policy: LogitsRowPolicy::Any,
            },
            2,
            &|_| DecoderKvPageStatus::Vacant,
        )
        .unwrap()
    }

    #[test]
    fn external_authority_rejects_unknown_id_generation_phase_and_terminal_state() {
        let transaction = ExecutionTransactionId::new(10).unwrap();
        let mut state = GenericDecoderSequenceState::new((), ());
        let batch = packed(&state, ForwardPhase::Prefill, 1);
        let mut active = ExpertForward {
            transaction,
            batch: batch.clone(),
            phase: ExpertForwardPhase::Active,
        };
        active.check(transaction, &batch).unwrap();
        assert!(
            active
                .check(ExecutionTransactionId::new(11).unwrap(), &batch)
                .is_err()
        );
        assert!(
            active
                .check(transaction, &packed(&state, ForwardPhase::Prefill, 2))
                .is_err()
        );
        assert!(
            active
                .check(transaction, &packed(&state, ForwardPhase::Decode, 1))
                .is_err()
        );
        state.core_mut().reset();
        assert!(
            active
                .check(transaction, &packed(&state, ForwardPhase::Prefill, 1))
                .is_err()
        );
        for phase in [ExpertForwardPhase::Failed, ExpertForwardPhase::Draining] {
            active.phase = phase;
            assert!(active.check(transaction, &batch).is_err());
        }
    }

    struct CancellationProbe {
        calls: Rc<std::cell::Cell<usize>>,
        panic: bool,
    }
    impl crate::transformer::expert_parallel::RoutedSwiGluExecutor for CancellationProbe {
        fn routed_swiglu(
            &mut self,
            _: crate::transformer::expert_parallel::RoutedSwiGluRequest<'_>,
            _: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
        ) -> Result<OperatorProgress<Rows>> {
            panic!("CPU custody contract must not run a forward")
        }
    }
    impl crate::decoder::HybridCudaRoutedExecutor for CancellationProbe {
        fn drain(&mut self) -> Result<crate::decoder::WorkProgress> {
            Ok(crate::decoder::WorkProgress::Complete)
        }
        fn shutdown(&mut self) -> Result<crate::decoder::WorkProgress> {
            Ok(crate::decoder::WorkProgress::Complete)
        }
        fn on_error(&mut self, _: ExecutionTransactionId) {
            self.calls.set(self.calls.get() + 1);
            assert!(!self.panic, "injected cancellation callback panic");
        }
    }
    fn custody(calls: Rc<std::cell::Cell<usize>>, panic: bool) -> ExternalExpertCustody {
        let state = GenericDecoderSequenceState::new((), ());
        ExternalExpertCustody {
            attachment: Some(HybridCudaRoutedExperts::new(
                ParallelRankId::new(7),
                Box::new(CancellationProbe { calls, panic }),
            )),
            forward: Some(ExpertForward {
                transaction: ExecutionTransactionId::new(10).unwrap(),
                batch: packed(&state, ForwardPhase::Prefill, 1),
                phase: ExpertForwardPhase::Active,
            }),
            ..Default::default()
        }
    }
    #[test]
    fn external_custody_notifies_failure_once_and_retains_forward_until_proof() {
        let calls = Rc::new(std::cell::Cell::new(0));
        let mut custody = custody(calls.clone(), false);
        custody.fail().unwrap();
        custody.fail().unwrap();
        custody.begin_drain().unwrap();
        custody.begin_drain().unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(
            custody.forward.as_ref().unwrap().phase,
            ExpertForwardPhase::Draining
        );
        custody.finish();
        assert!(custody.forward.is_none());
    }
    #[test]
    fn external_custody_callback_panic_does_not_replace_primary_failure() {
        let calls = Rc::new(std::cell::Cell::new(0));
        let mut custody = custody(calls.clone(), true);
        let error = Error::with_cleanup(
            "forward",
            cuda_error("primary forward error"),
            custody.fail(),
        );
        let text = error.to_string();
        assert!(text.contains("primary forward error"), "{text}");
        assert!(text.contains("completion unknown"), "{text}");
        assert_eq!(
            custody.forward.as_ref().unwrap().phase,
            ExpertForwardPhase::Failed
        );
        custody.fail().unwrap();
        assert_eq!(calls.get(), 1);
    }
}
