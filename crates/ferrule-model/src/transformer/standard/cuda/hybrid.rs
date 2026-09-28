//! CUDA protocol adapter over the one standard hybrid layer/output composition.
use super::super::{PreparedStandardLayer, PreparedStandardOutput, StandardTransformerHidden};

use super::super::RoutedLayerExecution;
use super::*;
use crate::decoder::{
    CudaHybridSequenceState, CudaKvView, DecoderLogits, DecoderTransactionContext,
    HybridCudaDevice, HybridCudaExpertProgress, HybridCudaRoutedExperts, HybridLayerSchema,
    HybridStateSchema, NoTerminalGuard, PackedDecoderBatch, StandardSequenceState,
};
use crate::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use crate::transformer::{
    Attention, BoundDecoderResources, LayerRequest, MemoryLayerWeightCache, NoPostLayerTap, Poll,
    PreparedTransformer, StateDictMaterializer, Step, TransformerModule,
};
use std::convert::Infallible;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertForwardPhase {
    Active,
    Failed,
    Draining,
}

#[derive(Debug)]
struct ExpertForward {
    transaction: ExecutionTransactionId,
    batch: PackedDecoderBatch,
    phase: ExpertForwardPhase,
}
impl ExpertForward {
    fn check(&self, transaction: ExecutionTransactionId, batch: &PackedDecoderBatch) -> Result<()> {
        if self.transaction != transaction
            || self.phase != ExpertForwardPhase::Active
            || &self.batch != batch
        {
            return Err(cuda_error(
                "external expert transaction/batch/generation/phase is not active",
            ));
        }
        Ok(())
    }
}

/// Freeze stable sequence identities in packed ROW order, not state-slot order.
fn routed_sequences(batch: &PackedDecoderBatch) -> Result<Vec<u64>> {
    if batch.row_to_sequence().len() != batch.len() {
        return Err(cuda_error("external expert packed row count mismatch"));
    }
    batch
        .row_to_sequence()
        .iter()
        .enumerate()
        .map(|(row, &index)| {
            let sequence = batch
                .sequences()
                .get(index)
                .filter(|sequence| sequence.query().contains(&row))
                .ok_or_else(|| cuda_error("external expert packed sequence mapping mismatch"))?;
            Ok(sequence.topology_id().get())
        })
        .collect()
}

struct RoutedCall<'a> {
    executor: &'a mut dyn crate::decoder::HybridCudaRoutedExecutor,
    device: &'a HybridCudaDevice,
    holds: &'a mut Vec<DeviceBuffer<f32>>,
}
impl RoutedCall<'_> {
    fn hold(&mut self, rows: &Rows) -> Result<()> {
        let rows = rows.cuda()?;
        rows.validate_owner(self.device.operators())?;
        let buffer = rows
            .f32_buffer()
            .ok_or_else(|| cuda_error("external expert requires F32 CUDA rows"))?;
        self.holds.push(
            buffer
                .as_device_buffer()
                .slice(0, rows.shape().elements())?,
        );
        Ok(())
    }
}
impl RoutedSwiGluExecutor for RoutedCall<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        if request.sequences.len() != request.input.shape().rows()
            || request.routes.rows() != request.sequences.len()
        {
            return Err(cuda_error("external routed rows/identities mismatch"));
        }
        let transaction = request.context.transaction;
        let shape = request.input.shape();
        let arena = request.arena;
        // Retain aliases BEFORE submission, including failure/panic paths. A root
        // stream fence alone says nothing about a remote consumer of these rows.
        self.hold(request.input)?;
        let result = self.executor.routed_swiglu(request, check_active)?;
        if let OperatorProgress::Ready(rows) = &result {
            self.hold(rows)?;
            if rows.shape() != shape || rows.arena() != arena {
                return Err(cuda_error("external expert result geometry mismatch"));
            }
        }
        check_active(transaction)?;
        Ok(result)
    }
}

pub struct CudaHybridModule {
    resources: Arc<BoundDecoderResources>,
    materializer: Arc<StateDictMaterializer>,
    device: HybridCudaDevice,
    schema: HybridStateSchema,
    experts: Option<HybridCudaRoutedExperts>,
    expert_forward: Option<ExpertForward>,
    expert_holds: Vec<DeviceBuffer<f32>>,
    expert_kv: Option<CudaKvView>,
    closed: bool,
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
            .field("experts", &self.experts)
            .field("expert_forward", &self.expert_forward)
            .field("expert_holds", &self.expert_holds.len())
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
        options.validate_hybrid_cuda(&resources)?;
        let spec = resources.spec();
        let active_layers = options.active_layers().unwrap_or(spec.layers().len());

        let schema = HybridStateSchema::from_spec(spec, active_layers)?;
        super::super::validate_hybrid_descriptors(spec, spec.layers())?;
        for layer in spec.layers() {
            if let Attention::Gqa(g) = layer.attention() {
                validate_rope(g.rotary())?;
            }
        }

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
        let result = operators.prepare_hybrid_image(&embedding, &layers, &output);
        if operators.needs_quarantine() {
            device.quarantine();
        }
        result?;
        if operators.resident_parameter_bytes() > device.budget().weight_bytes {
            return Err(unsupported("resident CUDA weights exceed budget"));
        }
        Ok(Self {
            resources,
            materializer,
            device,
            schema,
            experts,
            expert_forward: None,
            expert_holds: Vec::new(),
            expert_kv: None,
            closed: false,
            #[cfg(test)]
            reject_next_finish_completion: std::cell::Cell::new(false),
            operators,
            prepared: PreparedTransformer::new(embedding, layers, output, NoPostLayerTap),
        })
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
        self.device.ensure_ready()?;
        if let Some(experts) = &mut self.experts {
            self.expert_forward
                .as_ref()
                .ok_or_else(|| cuda_error("unknown external expert forward"))?
                .check(transaction, batch)?;
            if let Some(check) = &mut experts.check_active {
                check(transaction)?;
            }
        }
        Ok(())
    }

    fn expert_failed(&mut self) {
        if let Some(forward) = &mut self.expert_forward {
            if forward.phase == ExpertForwardPhase::Active {
                forward.phase = ExpertForwardPhase::Failed;
                if let Some(experts) = &mut self.experts {
                    experts
                        .executor
                        .as_mut()
                        .expect("attached executor")
                        .on_error(forward.transaction);
                }
            }
        }
    }

    fn drain_experts(&mut self, shutdown: bool) -> Result<bool> {
        self.device.ensure_ready()?;
        let Some(experts) = &mut self.experts else {
            return Ok(true);
        };
        let executor = experts.executor.as_mut().expect("attached executor");
        let progress = executor.drain()?;
        let progress = if progress == HybridCudaExpertProgress::Complete && shutdown {
            executor.shutdown()?
        } else {
            progress
        };
        match progress {
            HybridCudaExpertProgress::Complete => {
                self.device.fence()?;
                self.expert_holds.clear();
                Ok(true)
            }
            HybridCudaExpertProgress::Pending => Ok(false),
            HybridCudaExpertProgress::Unknown => {
                self.device.quarantine();
                self.operators.poisoned = true;
                self.device.ensure_ready()?;
                unreachable!()
            }
        }
    }

    fn require_expert_drain(&mut self, shutdown: bool) -> Result<()> {
        if !self.drain_experts(shutdown)? {
            return Err(cuda_error(
                "external expert completion pending; retain module/KV custody",
            ));
        }
        Ok(())
    }

    fn checked<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.device.ensure_ready()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = run(self);
            if result.is_err() {
                self.expert_failed();
            }
            result
        }));
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        match result {
            Ok(result) => result,
            Err(panic) => {
                self.device.quarantine();
                self.operators.poisoned = true;
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
            let forward = this.expert_forward.as_ref();
            let device = &this.device;
            let (source_rank, executor, mut external_check) = match this.experts.as_mut() {
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
                holds: &mut this.expert_holds,
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
            let mut binding = CudaStandardKvBinding::new(kv, batch)?;
            let mut states = states
                .iter_mut()
                .map(|s| s as &mut dyn StandardSequenceState)
                .collect::<Vec<_>>();
            let result = super::super::execute_standard_layer_inner(
                &mut this.operators,
                &this.materializer,
                layer.index(),
                layer.attention().block().as_ref(),
                layer.feed_forward().block(),
                kv_layer,
                hidden,
                &mut binding,
                &mut states,
                execution.as_mut(),
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
        self.device.ensure_ready()?;
        if self.closed || self.expert_forward.is_some() {
            return Err(cuda_error(
                "hybrid module closed or previous expert forward retains custody",
            ));
        }
        if context.transaction() != kv.transaction() {
            return Err(cuda_error("hybrid context/root KV transaction mismatch"));
        }
        batch.validate_states(states)?;
        routed_sequences(batch)?;
        self.operators.preflight_expert_metadata()?;
        for state in states {
            state.validate_standard_state(&self.schema)?;
        }
        kv.validate_batch(batch)
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
            if this.experts.is_some() {
                this.expert_forward = Some(ExpertForward {
                    transaction: context.transaction(),
                    batch: batch.clone(),
                    phase: ExpertForwardPhase::Active,
                });
                this.expert_kv = Some(kv.clone());
            }
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
            this.expert_failed();
            if let Some(forward) = &mut this.expert_forward {
                forward.phase = ExpertForwardPhase::Draining;
            }
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
        self.expert_forward = None;
        self.expert_kv = None;
        match events.into_iter().next() {
            Some(event) => match event {},
            None => Ok(NoTerminalGuard),
        }
    }

    fn abort(
        &mut self,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
    ) -> Result<()> {
        self.checked(|this| {
            this.require_expert_drain(false)?;
            this.device.fence()?;
            this.expert_forward = None;
            this.expert_kv = None;
            Ok(())
        })
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.closed {
            return self.device.ensure_ready();
        }
        self.checked(|this| {
            this.require_expert_drain(true)?;
            this.operators.quiesce()?;
            this.device.fence()?;
            if let Some(experts) = &mut this.experts {
                experts.closed = true;
            }
            this.expert_kv = None;
            this.expert_forward = None;
            this.closed = true;
            Ok(())
        })
    }
}

impl Drop for CudaHybridModule {
    fn drop(&mut self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.shutdown()));
        if !matches!(result, Ok(Ok(()))) || self.device.needs_quarantine() {
            self.device.quarantine();
            self.operators.poisoned = true;
            std::mem::forget(self.experts.take());
            std::mem::forget(self.expert_kv.take());
            std::mem::forget(std::mem::take(&mut self.expert_holds));
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
}
